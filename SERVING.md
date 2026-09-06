# Serving Layer - API and Demo UI

This document covers the application that serves the trained model. The modelling work,
evaluation and ethics discussion live in [README.md](README.md); this file exists so that
the serving detail does not crowd out the academic argument there.

The demo is not decoration. Two rubric points are only demonstrable with a running system:
the **deployment discussion** needs something real to anchor it, and the **cold-start
claim** is far more convincing when a marker can build a playlist the model has never seen
and watch it work.

---

## Why the model can be served without PyTorch

The item tower is deterministic at inference: given a track index and its audio features it
returns the same vector every time. The notebook therefore applies it to the entire
catalogue once and exports the result, so the API never loads PyTorch, the 1.4 GB training
cache, or the SQLite database.

Every request reduces to two operations:

1. **Fold-in** - average the seed tracks' item vectors to synthesise a user vector.
2. **Score** - one matrix-vector product over `float32` arrays, then a top-k.

For 81,738 tracks at 32 dimensions this is a 10 MB dot product: a few milliseconds, single
threaded, no GPU. The same property is what would allow the item vectors to move into an
ANN index (FAISS, HNSW) if the catalogue outgrew a dense scan.

> **The subtle part.** The synthetic user vector must be built from **post-fusion** item
> vectors, not raw ID embeddings. Those are two different vector spaces, and averaging the
> wrong one returns rankings that look plausible but are arbitrary. This was a real bug
> during development, and it is silent - nothing crashes, the output is just wrong.

---

## The deployment bundle

Written by the export cells of `main.ipynb` into `deploy/` (git-ignored, ~63 MB):

| File | Contents |
|---|---|
| `item_vectors.npy` | `(81738, 32)` float32 - every servable track |
| `user_vectors.npy` | `(328961, 32)` float32 - learned playlist embeddings |
| `catalogue.json` | Track id, name, artist list, genre list, plays, placement tier |
| `playlist_ids.json` | Index → original playlist id, for the warm path |
| `train_interactions.npz` | Sparse matrix used to exclude tracks a playlist already has |
| `serving_meta.json` | The trained/cold boundary and the measured cold-item scores |

**Two invariants the API validates at startup**, because a mismatched artefact loads
silently and produces plausible but wrong output:

- `catalogue.json` and `item_vectors.npy` must agree on catalogue size.
- `train_interactions.npz` must have exactly `trained_items` columns.

The first `22,748` rows of `item_vectors.npy` are the trained catalogue **in their original
training order**. Cold tracks are appended after them. This is load-bearing: it is what
keeps `user_vectors.npy` and `train_interactions.npz` aligned without remapping anything.

---

## Placement tiers

Only 22,748 of the 81,738 served tracks were trained on. The rest have no interaction
history at all and are positioned by proxy. Each track carries the tier that produced its
vector, and the API returns it with every result.

| Tier | Tracks | Vector source | Measured NDCG@10 |
|---|---|---|---|
| `learned` | 22,748 | Its own playlist co-occurrences | 0.7724 |
| `artist` | 25,766 | Mean of the artist's known tracks | 0.6554 |
| `genre` | 33,224 | Its genre centroid | 0.4441 |
| `catalogue` | 0 | Global centroid, last resort | - |

For reference, a global popularity ranker scores 0.3766 on the same task, so even the
weakest tier beats the trivial baseline. The methodology behind these numbers is in
[README.md § Placing a track the model has never seen](README.md#placing-a-track-the-model-has-never-seen).

Surfacing the tier in the UI is a deliberate explainability choice: a recommendation can be
challenged on the evidence behind it rather than accepted on trust.

---

## API reference

Base URL `http://127.0.0.1:8000`. CORS is restricted to the Vite dev server origins.

### `GET /api/health`
Catalogue size, trained-track count, per-tier breakdown, embedding dimension, and the
measured cold-item scores.

### `GET /api/genres`
All 114 genres with the number of artists carrying each. Alphabetical - ranking by artist
count buries `rock` behind niche genres full of single-track artists.

### `GET /api/artists`
| Param | Default | Meaning |
|---|---|---|
| `q` | `""` | Prefix matches rank above substring matches |
| `genres` | `""` | Comma-separated, **any-of** semantics |
| `sort` | `plays` | `plays` or `name` |
| `limit` / `offset` | `50` / `0` | Pagination for the infinite-scroll list |

Both orderings are precomputed at startup, so paging is stable across requests and the
search partition preserves whichever order is active. An artist inherits every genre any of
their tracks carries, so filtering by `rock` keeps a band only partly tagged as rock.

### `GET /api/artists/tracks?name=`
Every track by one artist, most played first, each with its placement tier.

### `GET /api/tracks?q=`
Token-AND search over `name + artists`, so `so what davis` and `davis so what` find the same
track. Exact-prefix title matches rank first, then by plays.

### `POST /api/recommend/new`
```json
{ "track_ids": ["...", "..."], "k": 10 }
```
The cold-start path. Returns the resolved seeds (with their tiers), any unrecognised ids,
and the ranked continuations. Seed tracks are excluded from their own recommendations.

### `GET /api/recommend/{playlist_id}`
The warm path, for a playlist that has a learned embedding. Tracks already in the playlist
are excluded using `train_interactions.npz`.

---

## The interface

Three panels, left to right, so the flow is linear.

**1. Pick an artist.** Genre chips (multi-select, any-of), a `Popular` / `A-Z` sort toggle,
a debounced search box, and an infinite-scroll list that pages 50 at a time.

**2. Add tracks.** Either the selected artist's tracks, or - with its own search box - any
track in the catalogue without going through an artist first. Clearing the search returns to
the artist that was being browsed.

**3. Your playlist.** The seeds, and the button that requests continuations.

Every track in panels 2 and 3 and in the results carries its tier badge, with the full
reason in the tooltip.

### UI decisions that were changed after testing them

- **Artist ordering.** Alphabetical was the initial request, but a raw ASCII sort opens on
  `!nvite`, `"Cats" 1983 Broadway Cast` and `#Kids`. Names starting with punctuation are now
  grouped last, and popularity is the default order with alphabetical available on the
  toggle - a first screen of `0-600` and `04 Limited Sazabys` is not a usable entry point.
- **Search ranking.** Sorting matches purely alphabetically put Green Bicycle above Green
  Day. Within each match group, plays now decide.
- **Genre chip ordering.** Ranking by artist count put `deep-house` and `show-tunes` first
  and pushed `rock` to position ~90, because the dataset holds 1,000 tracks per genre and
  niche genres accumulate single-track artists. Alphabetical is predictable.

---

## Running it

The `deploy/` bundle must exist first; it is produced by the export cells at the end of
`main.ipynb`.

```bash
# API - http://127.0.0.1:8000
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000

# UI - http://localhost:5173, proxies /api to the API
cd "frontend/Music Recommender" && npm install && npm run dev
```

`--reload` watches Python sources only. Re-exporting the bundle from the notebook does **not**
trigger a reload, so the API must be restarted manually after any export.

---

## A demo worth recording

Build a playlist from **only** these three blink-182 tracks, all badged `unseen · via artist`
because all three were released after the MPD dump was collected:

*EDGING*, *Darkside*, *I Really Wish I Hated You*

The continuations are MxPx, Pennywise, The Swellers, Millencolin and Masked Intruder - the
punk neighbourhood, reached without a single observed interaction for any seed. Repeating the
exercise with Metallica returns Iron Maiden and Slayer rather than punk, which shows the
conditioning is real and not a popularity fallback.

---

## Known limitations

- **Collaborations skew towards the more prolific artist.** The artist proxy weights by
  track count, so a blink-182 × XXXTENTACION track lands in emo-rap. Measured against
  one-vote-per-artist the difference is negligible in aggregate (0.6554 vs 0.6546), so the
  simpler rule was kept - but it is visible on individual collaborations.
- **Duplicate recordings survive in the trained catalogue.** The same song exists under
  several Spotify ids. Cold duplicates are removed, but trained ones are kept because their
  indices are load-bearing.
- **No authentication, rate limiting or persistence.** This is a local demonstrator for an
  academic submission, not a deployable service. The deployment discussion in the README
  addresses what production would additionally require.
