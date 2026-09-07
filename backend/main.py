"""Inference API for the Two-Tower playlist-continuation model.

The notebook exports a deployment bundle in which the item tower has already been applied
to the whole catalogue, so serving needs neither PyTorch nor the 1.4 GB training cache: a
request is a mean over seed vectors followed by one matrix-vector product and a top-k.
"""

import json
import os
import base64
import urllib.request
import urllib.parse
import urllib.error
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.sparse import load_npz
from dotenv import load_dotenv

load_dotenv()

DEPLOY = Path(__file__).resolve().parent / "deploy"

state: dict = {}


class NewPlaylistRequest(BaseModel):
    track_ids: list[str] = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)
    exclude_artists: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    required = (
        "item_vectors.npy",
        "user_vectors.npy",
        "catalogue.json",
        "playlist_ids.json",
        "train_interactions.npz",
    )
    missing = [f for f in required if not (DEPLOY / f).is_file()]
    if missing:
        raise RuntimeError(
            f"Missing deployment artefacts: {missing}. Run the export cell in main.ipynb."
        )

    item_vectors = np.load(DEPLOY / "item_vectors.npy")
    catalogue = json.loads((DEPLOY / "catalogue.json").read_text(encoding="utf-8"))
    if len(catalogue) != item_vectors.shape[0]:
        raise RuntimeError("catalogue.json and item_vectors.npy disagree on catalogue size.")

    state["item_vectors"] = item_vectors
    state["user_vectors"] = np.load(DEPLOY / "user_vectors.npy")
    state["catalogue"] = catalogue
    state["index_by_track"] = {row["track_id"]: i for i, row in enumerate(catalogue)}
    state["playlist_ids"] = json.loads((DEPLOY / "playlist_ids.json").read_text())
    state["index_by_playlist"] = {p: i for i, p in enumerate(state["playlist_ids"])}
    state["history"] = load_npz(DEPLOY / "train_interactions.npz").tocsr()
    state["plays"] = np.array([row["plays"] for row in catalogue])

    meta_file = DEPLOY / "serving_meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
    # Tracks at or beyond this index were never trained on; they are placed by proxy.
    state["trained_items"] = int(meta.get("trained_items", item_vectors.shape[0]))
    state["cold_item_ndcg"] = meta.get("cold_item_ndcg", {})
    if state["history"].shape[1] != state["trained_items"]:
        raise RuntimeError("train_interactions.npz does not match the trained item count.")

    # Artist index, so the catalogue can be browsed rather than guessed at. Built from the
    # exported list rather than by splitting the display string, which would corrupt names
    # containing commas.
    artists: dict[str, dict] = {}
    for i, row in enumerate(catalogue):
        for artist in row.get("artist_list") or []:
            key = artist.strip().lower()
            if not key:
                continue
            entry = artists.setdefault(
                key, {"name": artist.strip(), "tracks": [], "plays": 0, "genres": set()}
            )
            entry["tracks"].append(i)
            entry["plays"] += int(row["plays"])
            # An artist inherits every genre any of their tracks carries, so filtering by
            # "rock" keeps a band that is only partly tagged as rock.
            entry["genres"].update(row.get("genre_list") or [])
    for entry in artists.values():
        entry["tracks"].sort(key=lambda i: -state["plays"][i])
    state["artists"] = artists
    # Alphabetical order is fixed once here so paging through it stays stable between
    # requests. Names opening with punctuation ("!nvite", "#Kids") sort before letters in
    # raw ASCII, which would fill the first screen with noise, so they are grouped last.
    # Both orderings are fixed once here so paging through either stays stable between
    # requests. Names opening with punctuation ("!nvite", "#Kids") sort before letters in
    # raw ASCII, which would fill the first screen with noise, so they are grouped last.
    state["artist_keys_by_name"] = sorted(
        artists, key=lambda k: (0 if k[:1].isalnum() else 1, k)
    )
    state["artist_keys_by_plays"] = sorted(
        artists, key=lambda k: (-artists[k]["plays"], k)
    )

    genre_artists: dict[str, int] = {}
    for entry in artists.values():
        for genre in entry["genres"]:
            genre_artists[genre] = genre_artists.get(genre, 0) + 1
    state["genre_artists"] = genre_artists

    # Lowercased once so the picker's substring search does not rebuild it per request.
    state["haystack"] = [f"{row['name']} {row['artists']}".lower() for row in catalogue]
    state["names_lower"] = [row["name"].lower() for row in catalogue]
    yield
    state.clear()


app = FastAPI(title="Playlist Continuation API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# How each tier's vector was obtained, surfaced so a recommendation can be challenged on
# the basis of the evidence behind it rather than taken on trust.
TIER_REASON = {
    "learned": "learned from its own playlist co-occurrences",
    "artist": "never observed; placed by averaging its artist's known tracks",
    "genre": "never observed and artist unknown; placed by its genre centroid",
    "catalogue": "no usable signal; placed at the catalogue centroid",
}


def _entry(index: int, score: float | None = None) -> dict:
    row = state["catalogue"][index]
    out = {k: row[k] for k in ("track_id", "name", "artists", "genres")}
    tier = row.get("tier", "learned")
    out["tier"] = tier
    out["placement"] = TIER_REASON.get(tier, tier)
    if score is not None:
        out["score"] = round(float(score), 4)
    return out


def _rank(scores: np.ndarray, exclude: set, k: int) -> list[dict]:
    n = min(len(scores), k + len(exclude))
    top = np.argpartition(-scores, n - 1)[:n]
    top = top[np.argsort(-scores[top])]
    return [_entry(int(i), scores[i]) for i in top if int(i) not in exclude][:k]


@app.get("/api/health")
def health():
    tiers: dict[str, int] = {}
    for row in state["catalogue"]:
        tier = row.get("tier", "learned")
        tiers[tier] = tiers.get(tier, 0) + 1
    return {
        "status": "ok",
        "tracks": len(state["catalogue"]),
        "trained_tracks": state["trained_items"],
        "tiers": tiers,
        "artists": len(state["artists"]),
        "playlists": len(state["playlist_ids"]),
        "embedding_dim": int(state["item_vectors"].shape[1]),
        "cold_item_ndcg": state["cold_item_ndcg"],
    }


@app.get("/api/genres")
def list_genres():
    """Genres available as filters. Alphabetical, because ranking them by artist count
    buries "rock" behind niche genres full of one-track artists."""
    counts = state["genre_artists"]
    return {
        "results": [
            {"name": g, "artist_count": counts[g]} for g in sorted(counts)
        ]
    }


@app.get("/api/artists")
def list_artists(
    q: str = Query(default="", max_length=100),
    genres: str = Query(default="", max_length=500),
    sort: str = Query(default="plays", pattern="^(plays|name)$"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Artists present in the catalogue, so the tester never has to guess what exists."""
    needle = q.strip().lower()
    artists = state["artists"]
    keys = state["artist_keys_by_plays" if sort == "plays" else "artist_keys_by_name"]

    wanted = {g.strip().lower() for g in genres.split(",") if g.strip()}
    if wanted:
        # Any-of rather than all-of: an artist tagged rock and grunge survives a rock filter.
        keys = [k for k in keys if artists[k]["genres"] & wanted]

    if needle:
        # Starts-with wins over contains, so typing "sum" offers "Sum 41" before "Possum".
        # Partitioning preserves the chosen ordering inside each group.
        prefix = [k for k in keys if k.startswith(needle)]
        keys = prefix + [k for k in keys if needle in k and not k.startswith(needle)]

    page = keys[offset : offset + limit]
    return {
        "total": len(keys),
        "offset": offset,
        "has_more": offset + len(page) < len(keys),
        "results": [
            {"name": artists[k]["name"], "track_count": len(artists[k]["tracks"])}
            for k in page
        ],
    }


@app.get("/api/artists/tracks")
def artist_tracks(name: str = Query(min_length=1), limit: int = Query(50, ge=1, le=200)):
    entry = state["artists"].get(name.strip().lower())
    if entry is None:
        raise HTTPException(status_code=404, detail=f"'{name}' is not in the catalogue.")
    return {
        "artist": entry["name"],
        "total": len(entry["tracks"]),
        "results": [_entry(i) for i in entry["tracks"][:limit]],
    }


@app.get("/api/tracks")
def search_tracks(
    q: str = Query(default="", max_length=100), limit: int = Query(20, ge=1, le=50)
):
    """Substring search over the catalogue, so a playlist can be built without knowing IDs."""
    needle = q.strip().lower()
    if not needle:
        order = np.argsort(-state["plays"])[:limit]
        return {"query": q, "total": len(state["catalogue"]), "results": [_entry(int(i)) for i in order]}

    # Every token must appear somewhere in "name + artists", so word order does not matter
    # and "davis so what" finds the same track as "so what davis".
    tokens = needle.split()
    hits = [i for i, hay in enumerate(state["haystack"]) if all(t in hay for t in tokens)]
    names = state["names_lower"]
    hits.sort(key=lambda i: (0 if names[i].startswith(needle) else 1, -state["plays"][i]))
    return {"query": q, "total": len(hits), "results": [_entry(i) for i in hits[:limit]]}


@app.post("/api/recommend/new")
def recommend_new(request: NewPlaylistRequest):
    """Fold-in for a playlist the model has never seen."""
    index_by_track = state["index_by_track"]
    seeds = [index_by_track[t] for t in request.track_ids if t in index_by_track]
    if not seeds:
        raise HTTPException(
            status_code=400, detail="None of the supplied tracks are in the catalogue."
        )

    item_vectors = state["item_vectors"]
    # Averaged over POST-fusion item vectors: that is the space the user embeddings were
    # trained to dot against. Averaging raw ID embeddings would mix two different spaces.
    synthetic_user = item_vectors[seeds].mean(axis=0)
    scores = item_vectors @ synthetic_user
    
    exclude = set(seeds)
    if request.exclude_artists:
        artists_to_exclude = set()
        for s in seeds:
            row = state["catalogue"][s]
            for artist in row.get("artist_list") or []:
                artists_to_exclude.add(artist.strip().lower())
        for artist in artists_to_exclude:
            if artist in state["artists"]:
                exclude.update(state["artists"][artist]["tracks"])

    return {
        "seeds": [_entry(i) for i in seeds],
        "unknown_tracks": [t for t in request.track_ids if t not in index_by_track],
        "recommendations": _rank(scores, exclude, request.k),
    }


@app.get("/api/recommend/{playlist_id}")
def recommend_historical(playlist_id: str, k: int = Query(10, ge=1, le=100)):
    """Warm path: the playlist already has a learned embedding."""
    u = state["index_by_playlist"].get(playlist_id)
    if u is None:
        raise HTTPException(
            status_code=404, detail="Playlist not present in the training data."
        )

    history = state["history"]
    known = set(history.indices[history.indptr[u] : history.indptr[u + 1]].tolist())
    scores = state["item_vectors"] @ state["user_vectors"][u]

    return {
        "playlist_id": playlist_id,
        "known_tracks": len(known),
        "recommendations": _rank(scores, known, k),
    }

spotify_token = {"access_token": "", "expires_at": 0}

def get_spotify_token():
    now = time.time()
    if spotify_token["access_token"] and now < spotify_token["expires_at"]:
        return spotify_token["access_token"], None
    
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    if not client_id or not client_secret:
        return "", "Missing SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET in environment"
    
    auth_str = f"{client_id}:{client_secret}"
    b64_auth_str = base64.b64encode(auth_str.encode()).decode()
    
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token",
        data=urllib.parse.urlencode({"grant_type": "client_credentials"}).encode(),
        headers={
            "Authorization": f"Basic {b64_auth_str}",
            "Content-Type": "application/x-www-form-urlencoded"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            spotify_token["access_token"] = res_data["access_token"]
            spotify_token["expires_at"] = now + res_data.get("expires_in", 3600) - 60
            return spotify_token["access_token"], None
    except urllib.error.URLError as e:
        error_msg = str(e)
        if hasattr(e, 'read'):
            error_msg += " " + e.read().decode()
        return "", f"Token Error: {error_msg}"

@app.get("/api/covers")
def get_covers(ids: str = Query(..., min_length=1)):
    """Fetch album covers from Spotify given a comma-separated list of track IDs (max 50)."""
    token, err = get_spotify_token()
    if not token:
        return {"data": {}, "error": err}
        
    track_ids = [t.strip().split(":")[-1] for t in ids.split(",") if t.strip()][:50]
    if not track_ids:
        return {"data": {}, "error": "No valid track IDs provided"}
        
    req = urllib.request.Request(
        f"https://api.spotify.com/v1/tracks?ids={','.join(track_ids)}",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    spotify_data = {}
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode())
            for track in res_data.get("tracks", []):
                if track:
                    t_id = track["id"]
                    spotify_data[t_id] = {}
                    if track.get("album") and track["album"].get("images"):
                        images = track["album"]["images"]
                        spotify_data[t_id]["cover"] = images[-1]["url"]
                    if track.get("preview_url"):
                        spotify_data[t_id]["preview_url"] = track["preview_url"]
    except urllib.error.URLError as e:
        error_msg = str(e)
        if hasattr(e, 'read'):
            error_msg += " " + e.read().decode()
        return {"data": spotify_data, "error": f"API Error: {error_msg}"}
        
    return {"data": spotify_data, "error": None}

