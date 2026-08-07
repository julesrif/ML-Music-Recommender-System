"""Stream a mysqldump schema + data dump into a local SQLite database.

Used to load the zenodo "Playlist2vec" MySQL dump (schema + ~10.7 GB of
`INSERT` statements) without needing a MySQL server or loading the whole
dump into memory at once.
"""

import re
import sqlite3
from pathlib import Path

INSERT_RE = re.compile(r"^INSERT INTO `(\w+)` VALUES (.*);\s*$")


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
    """Translate mysqldump CREATE TABLE statements into SQLite-compatible DDL."""
    statements = []
    for table_name, body in re.findall(
        r"CREATE TABLE `(\w+)` \((.*?)\) ENGINE", schema_sql, flags=re.S
    ):
        kept_lines = []
        for line in body.strip().splitlines():
            line = line.strip().rstrip(",")
            if re.match(r"^(KEY|UNIQUE KEY|CONSTRAINT)\b", line, flags=re.I):
                continue
            kept_lines.append(line)
        statements.append(
            f"CREATE TABLE IF NOT EXISTS `{table_name}` ({', '.join(kept_lines)})"
        )
    return statements


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
    for ddl in _schema_to_sqlite_ddl(schema_sql):
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

            placeholders = ",".join("?" * len(rows[0]))
            insert_stmt = f"INSERT INTO `{table_name}` VALUES ({placeholders})"
            for i in range(0, len(rows), batch_size):
                cur.executemany(insert_stmt, rows[i : i + batch_size])
            conn.commit()
            print(f"Loaded {len(rows)} rows into {table_name}")

    conn.close()
