"""Stream a mysqldump schema + data dump into a local SQLite database.

Used to load the zenodo "Playlist2vec" MySQL dump (schema + ~10.7 GB of
`INSERT` statements) without needing a MySQL server or loading the whole
dump into memory at once.
"""

import re
import sqlite3
from pathlib import Path

INSERT_RE = re.compile(r"^INSERT INTO `(\w+)` VALUES (.*);\s*$")

# Data minimisation. Two columns in `playlist` are stripped before anything reaches disk:
#   name -> user-generated free text leaking PII and special-category data (mood, religion,
#           sexual orientation).
#   uri  -> embeds the owner's Spotify user id ("spotify:user:<id>:playlist:<id>"), which is
#           a persistent, directly identifying handle; many are real names rather than
#           opaque ids, and they allow re-identification against public Spotify profiles.
# `track`/`album`/`artist` names and uris are public catalogue metadata and are kept.
# Verified against spotifydbdumpschemashare.sql (Zenodo 10.5281/zenodo.5002584).
REDACTED_COLUMNS = {
    "playlist": {"name", "uri"},
}


def _split_top_level(text, opener="(", closer=")"):
    """Split VALUES (...),(...),... into the raw content of each (...) group, quote-aware."""
    groups = []
    depth = 0
    in_string = False
    escape = False
    start = None
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_string = False
            continue
        if ch == "'":
            in_string = True
        elif ch == opener:
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                groups.append(text[start:i])
    return groups


def _split_fields(tuple_text):
    """Split one row's raw tuple content on top-level commas, respecting quoted strings."""
    fields = []
    buf = []
    in_string = False
    escape = False
    for ch in tuple_text:
        if in_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "'":
                in_string = False
            continue
        if ch == "'":
            in_string = True
            buf.append(ch)
        elif ch == ",":
            fields.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    fields.append("".join(buf))
    return fields


def _convert_field(raw):
    raw = raw.strip()
    if raw == "NULL":
        return None
    if raw.startswith("'") and raw.endswith("'"):
        inner = raw[1:-1]
        return (
            inner.replace("\\'", "'")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\\\", "\\")
        )
    try:
        return int(raw)
    except ValueError:
        return float(raw)


def _schema_to_sqlite_ddl(schema_sql):
    """Translate mysqldump CREATE TABLE statements into SQLite-compatible DDL.

    Also returns, per table, the ordered column names kept and their positions in the
    original dump, so redacted fields can be stripped from each INSERT tuple.
    """
    statements = []
    layout = {}
    for table_name, body in re.findall(
        r"CREATE TABLE `(\w+)` \((.*?)\) ENGINE", schema_sql, flags=re.S
    ):
        redacted = REDACTED_COLUMNS.get(table_name.lower(), set())
        kept_lines = []
        kept_columns = []
        kept_indices = []
        dropped = []
        position = 0

        for line in body.strip().splitlines():
            line = line.strip().rstrip(",")
            if re.match(r"^(KEY|UNIQUE KEY|CONSTRAINT)\b", line, flags=re.I):
                continue
            if re.match(r"^PRIMARY KEY\b", line, flags=re.I):
                if not any(f"`{col}`" in line for col in dropped):
                    kept_lines.append(line)
                continue

            column = re.match(r"^`(\w+)`", line)
            if not column:
                continue
            name = column.group(1)
            if name.lower() in redacted:
                dropped.append(name)
            else:
                kept_lines.append(line)
                kept_columns.append(name)
                kept_indices.append(position)
            position += 1

        if redacted and not dropped:
            print(
                f"WARNING: no redactable column found in `{table_name}`; "
                f"check the dump schema against REDACTED_COLUMNS."
            )
        for name in dropped:
            print(f"Redacting PII column {table_name}.{name} (never written to disk)")

        layout[table_name] = {
            "columns": kept_columns,
            "indices": kept_indices,
            "width": position,
            "redacted": bool(dropped),
        }
        statements.append(
            f"CREATE TABLE IF NOT EXISTS `{table_name}` ({', '.join(kept_lines)})"
        )
    return statements, layout


def load_mysql_dump_to_sqlite(schema_path, data_path, db_path, batch_size=5000):
    """Build a SQLite database at db_path from a mysqldump schema_path + data_path (idempotent)."""
    if Path(db_path).exists():
        print(f"{db_path} already exists, skipping import.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = MEMORY")

    schema_sql = Path(schema_path).read_text(encoding="utf-8", errors="ignore")
    ddl_statements, layout = _schema_to_sqlite_ddl(schema_sql)
    for ddl in ddl_statements:
        cur.execute(ddl)
    conn.commit()

    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = INSERT_RE.match(line.strip())
            if not match:
                continue
            table_name, values_block = match.groups()
            rows = [
                [_convert_field(v) for v in _split_fields(raw_tuple)]
                for raw_tuple in _split_top_level(values_block)
            ]
            if not rows:
                continue

            table = layout.get(table_name)
            if table and table["redacted"]:
                if len(rows[0]) != table["width"]:
                    raise ValueError(
                        f"`{table_name}` rows have {len(rows[0])} fields but the schema "
                        f"declares {table['width']}; refusing to redact by position."
                    )
                rows = [[row[i] for i in table["indices"]] for row in rows]
                columns = ", ".join(f"`{c}`" for c in table["columns"])
                target = f"`{table_name}` ({columns})"
            else:
                target = f"`{table_name}`"

            placeholders = ",".join("?" * len(rows[0]))
            insert_stmt = f"INSERT INTO {target} VALUES ({placeholders})"
            for i in range(0, len(rows), batch_size):
                cur.executemany(insert_stmt, rows[i : i + batch_size])
            conn.commit()
            print(f"Loaded {len(rows)} rows into {table_name}")

    conn.close()
