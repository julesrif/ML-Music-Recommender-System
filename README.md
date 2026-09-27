# Spotify Playlist Continuation

**Module:** Comparative Intelligence and Ethics in Machine Learning Systems

**University of Essex Online**

## Abstract

Playlist continuation on implicit feedback: given the tracks already in a playlist, rank the
tracks most likely to be added next. Spotify audio features (Kaggle) are joined with
playlist-track membership from the Playlist2vec dump, and a classical matrix-factorisation
baseline (TruncatedSVD) is compared with a two-tower neural network under a leave-one-out
ranking protocol. The project also covers cold-start playlists and tracks, SHAP/LIME
explainability, exposure and subgroup diagnostics, and a numpy-only inference API with a
Vue demo.

The unit of analysis is the playlist, not the person: the owner handle present in the dump is
stripped at ingestion (see [Privacy and data minimisation](#privacy-and-data-minimisation)).

Results, figures and their interpretation live in [main.ipynb](main.ipynb); this README
describes the system and how to reproduce and serve it.

---

## Repository Layout

```
main.ipynb                   end-to-end analysis: data prep, training, evaluation, XAI, export
mysql_to_sqlite.py           streaming MySQL dump -> SQLite loader with PII column redaction
dataset/                     Kaggle CSV and Playlist2vec dump (downloaded) and spotify.db (built)
cache/                       cached training frames and evaluation results (generated)
models/                      trained PyTorch weights (generated)
deploy/                      serving bundle exported by the notebook (generated)
backend/                     FastAPI inference API
frontend/Music Recommender/  Vue 3 + Vite demo UI
```

Generated folders are not versioned; running the notebook recreates them.

---

## Datasets

| Dataset | Source | Licence | Role |
|---|---|---|---|
| **Spotify Tracks Dataset** | Kaggle (maharshipandya/-spotify-tracks-dataset) | [ODbL 1.0](http://opendatacommons.org/licenses/odbl/1.0/) | Per-track audio features (danceability, energy, tempo, key, valence, etc.) used as the content-side signal. |
| **Playlist2vec** | Zenodo (10.5281/zenodo.5002584) | CC BY 4.0 | Real playlist-to-track membership, used as the implicit collaborative-filtering signal. |

The two sources are joined on Spotify track id. Playlist2vec is a crawl of ~1M user-created
and Spotify-curated playlists collected through the Spotify Web API (Papreja et al., 2019),
shipped as a 10.7 GB MySQL dump.

### Licence and usage constraints

Playlist2vec is published under **CC BY 4.0**. That licence covers the authors' compilation,
not the underlying Spotify content or the consent of the playlist owners, so the project
runs under self-imposed constraints:

- **Academic use only** - a university proof of concept; not commercialised or offered to consumers.
- **No redistribution** - the raw database is never hosted or served; the API returns model outputs only.
- **Isolated demo** - ephemeral, restricted to reviewers, not search-indexed, no user-data collection.

### Privacy and data minimisation

- `playlist.name` (free text, often sensitive) and `playlist.uri` (embeds the owner's
  Spotify handle, `spotify:user:<id>:playlist:<id>`, in 100% of rows) are dropped inside
  [mysql_to_sqlite.py](mysql_to_sqlite.py) (`REDACTED_COLUMNS`), from both the DDL and every
  `INSERT` tuple, so they never reach disk. The loader aborts rather than falling back to
  positional inserts if a row's width does not match the declared schema.
- Training artefacts hold numeric dump ids, integer indices and scaled audio features only.
- Exported playlist ids are salted SHA-256 hashes (`SALT` in `.env`).
- The demo fetches album covers and 30-second previews from the public Deezer API through
  the backend; only track and artist names are sent.

This is data minimisation, not a claim of regulatory compliance.

---

## Pipeline

### 1. Data preparation

- **Kaggle:** one row per `(track, genre)` is aggregated into one row per track with a genre
  list; rows with `tempo == 0` are dropped and `time_signature == 0` is imputed with the mode.
- **Playlist2vec:** the dump is streamed into SQLite (`dataset/spotify.db`) once. The
  audio-feature join runs inside SQLite, backed by a one-off index on
  `track_playlist1(track_id)` (~3.8 GB), so only matching rows reach pandas.
- **Filtering:** playlists over 250 tracks are dropped as outliers, then an iterative
  5-core keeps playlists and tracks with at least 5 interactions.
- **Cold-start cohort:** playlists removed by the 5-core that keep at least 2 core tracks
  are held aside for fold-in evaluation.

| Stage | Interactions | Tracks | Playlists |
|---|---|---|---|
| `track_playlist1` as shipped | 125,451,006 | 13,282,009 | ~1.05M |
| Kaggle tracks with audio features | - | 89,584 | - |
| Joined on track id | 7,218,586 | 31,403 | - |
| Length cap + 5-core (training set) | 6,271,723 | 22,616 | 328,279 |
| Cold-start cohort | 552,761 | - | 196,687 |

Interaction-matrix density after filtering: 0.000845. These counts depend only on the source
data and the filters above, not on training, so they are stable across runs.

### 2. Models

A single 80/20 split over interactions (seed 3) is shared by every model. Audio features are
scaled by a `StandardScaler` fitted on training positives only, and the same scaler feeds
evaluation, explainability and the export. Two-Tower v2 further splits the training set
90/10 into fit/validation for early stopping.

| Model | Description |
|---|---|
| Global popularity | Training interaction count of the track |
| Genre popularity | Popularity × match between the track's genres and the playlist's genre profile |
| TruncatedSVD | 50 components on the binary playlist × track matrix |
| Two-Tower v1 | 32-d playlist and track ID embeddings; an audio MLP (BatchNorm) is fused into the item tower. BCE, uniform negatives, 10 epochs, lr 3e-4 |
| **Two-Tower v2** (served) | Same architecture. Popularity-proportional negatives ($p \propto f^{0.75}$), BCE with logits, lr 1e-3, batch 4096, early stopping on validation NDCG@10 (max 20 epochs, patience 3) |
| Two-Tower v2, no audio | ID embeddings only, BatchNorm + ReLU on the item embedding (ablation) |

### 3. Evaluation

- **Primary protocol:** leave-one-out, one held-out positive ranked against 99 sampled
  negatives the playlist never contained; Recall@k and NDCG@k at k = 5, 10, 20. All systems
  are scored on identical candidate sets.
- **Stability check:** 10-fold cross-validation of SVD under the same protocol.
- **Cold-start playlists:** folded in from their seed tracks (SVD: `row @ components_.T`;
  Two-Tower: mean of the seed tracks' item-tower vectors, the same path the API uses), then
  scored under the same protocol.
- **Exposure:** full-catalogue top-10 for 2,000 playlists, giving coverage@10, Gini of
  exposure, mean popularity percentile, a Lorenz curve and full-catalogue NDCG@10.
- **Genre coherence@10** under exact, behavioural (playlist co-occurrence) and acoustic
  (audio centroid) genre similarity.
- **Subgroup NDCG@10** by playlist size (≤ 10 vs > 10 training tracks).
- **Paired bootstrap** (1,000 resamples) of audio vs no-audio per-playlist NDCG@10.
- RMSE is printed as a diagnostic only; the two models' RMSEs are measured on different data
  and are not comparable.

### 4. Explainability

- **SHAP** `KernelExplainer` over the audio inputs of Two-Tower v2 with IDs held fixed
  (100 interactions, 50 k-means background centroids, 200 samples each).
- **LIME** `LimeTabularExplainer` (classification mode) on the same interaction, compared
  with SHAP by sign.
- **SVD factor attribution:** exact per-factor decomposition of the SVD score, each factor
  labelled by the genres its item loadings concentrate on.

### 5. Cold-item placement and export

- A track with no interactions is placed at the mean learned vector of its artist's trained
  tracks, falling back to its genre centroid, then to the catalogue mean. Duplicate
  recordings are removed.
- The export applies the item tower to the whole catalogue, so the API needs neither PyTorch
  nor the training cache. Trained tracks keep their training indices; cold tracks are
  appended after them, each tagged with the tier (`learned`, `artist`, `genre`) that placed it.

---

## Technology Stack

### Data & Environment

| Library | Version | Purpose |
|---|---|---|
| **pandas** | 3.0.5 | Data manipulation and numerical computation |
| **numpy** | 2.5.1 | Numerical computation, array operations |
| **matplotlib** | 3.11.1 | Visualisation (sparsity plots, SHAP/LIME plots) |
| **seaborn** | 0.13.2 | Statistical visualisation |
| **missingno** | 0.5.2 | Missing-data visualisation during initial inspection |
| **python-dotenv** | 1.2.2 | Environment variable loading (API keys) |
| **kaggle** | 2.2.4 | Programmatic download of the Spotify Tracks Dataset |
| **zenodo-get** | 3.0.3 | Programmatic download of the Playlist2vec dump from Zenodo |
| **ipykernel / ipython** | 7.3.0 / 9.16.1 | Notebook kernel and interactive execution |
| **pre-commit** | 4.6.1 | Git hook framework for code quality checks |
| sqlite3 / mysql_to_sqlite | stdlib / local module | Streaming conversion of the MySQL dump into a local SQLite database without full in-memory load |

### Modelling & Evaluation

| Library | Version | Purpose |
|---|---|---|
| **scipy.sparse** | (via scikit-learn stack) | Sparse interaction matrix construction |
| **scikit-learn** | 1.9.0 | Train/test split, `TruncatedSVD` baseline, `StandardScaler`, RMSE and NDCG metrics |
| **PyTorch** | 2.13.0 | Two-tower deep learning model (embeddings, dense audio tower, fusion layer, training loop) |

### Explainability

| Library | Version | Purpose |
|---|---|---|
| **SHAP** | 0.52.0 | Model-agnostic feature attribution for the DL model's audio-feature inputs |
| **LIME** | 0.2.0.1 | Local surrogate-model explanations, cross-validated against SHAP |

### Serving & Demo

The API has no deep-learning dependency: the item tower is applied once in the notebook and
only its output is shipped.

| Library | Purpose |
|---|---|
| **FastAPI / uvicorn** | Inference API over the exported bundle; numpy-only at request time |
| **numpy / scipy.sparse** | Scoring (`item_vectors @ user_vector`) and the interaction matrix |
| **Vue 3 / TypeScript** | Demo interface for assembling a playlist and reading back the tiers |
| **Vite** | Dev server and `/api` proxy to the backend |
| **Tailwind CSS** | Styling |

---


## Getting Started

### Prerequisites

- Python 3.13.5
- Kaggle API credentials, for the automatic dataset download
- ~11 GB for the Playlist2vec dump, plus the generated SQLite database and a ~3.8 GB index
- Node.js (frontend) and optionally Docker

### 1. Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

| Variable | Used by | Purpose |
|---|---|---|
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | notebook | Kaggle dataset download |
| `SALT` | notebook export | Salt for the hashed playlist ids in `deploy/playlist_ids.json` |

### 3. Run the notebook

Run [main.ipynb](main.ipynb) top to bottom. The first run downloads both datasets, builds
`dataset/spotify.db` and trains the models. Artefacts are trained-if-absent, else loaded:

| Artefact | Contents |
|---|---|
| `cache/dl_cache.pkl` | Two-Tower v1 training frames, scaler and index maps |
| `cache/loo_multi_k_metrics.pkl` | v1 / SVD leave-one-out metrics |
| `cache/two_tower_v2_cache.pkl` | Two-Tower v2 (audio and no-audio) weights and training histories |
| `cache/ablation_history.json`, `models/two_tower_v2_noaudio_fair.pth` | no-audio ablation |
| `models/two_tower_model.pth`, `models/two_tower_v2.pth` | Two-Tower v1 and v2 weights |
| `deploy/*` | serving bundle (see [Serving](#serving)) |

Delete an artefact to force it to be recomputed. Every stochastic step uses a fixed seed
(`RANDOM_SEED = 3`) and its own generator, SQLite results are sorted before use, and PyTorch
runs with deterministic algorithms, so a clean run reproduces the notebook's outputs on the
same software versions. A full run from an empty `cache/` and `models/` takes several hours on CPU.

---

## Serving

The last notebook cells export `deploy/`:

| File | Contents |
|---|---|
| `item_vectors.npy` | item-tower output for every served track (trained rows first, cold tracks appended) |
| `user_vectors.npy` | one learned 32-d embedding per training playlist |
| `train_interactions.npz` | training interaction matrix, used to mask known tracks |
| `catalogue.json` | track id, name, artists, genres, training plays and placement tier |
| `playlist_ids.json` | salted hashes of the training playlist ids |
| `serving_meta.json` | trained/total item counts and cold-item NDCG figures |

A request for a new playlist is a mean over the seed tracks' item vectors followed by one
matrix-vector product and a top-k.

### Run with Docker

```bash
docker compose up --build
```

UI on http://localhost:8075, API on http://localhost:8091. `./deploy` is mounted into the
backend container.

### Run locally

The backend reads its bundle from `backend/deploy/`, so link the exported folder first:

```powershell
# Windows
New-Item -ItemType Junction -Path backend\deploy -Target (Resolve-Path deploy)
# Linux / macOS: ln -s ../deploy backend/deploy

python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8091
```

```bash
cd "frontend/Music Recommender"
npm install
npm run dev          # http://localhost:5173, proxies /api to port 8091
```

### API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Catalogue size, tier counts, embedding dimension, cold-item NDCG |
| GET | `/api/genres` | Genres available as filters |
| GET | `/api/artists` | Paged artist search (`q`, `genres`, `sort=plays\|name`, `limit`, `offset`) |
| GET | `/api/artists/tracks` | Tracks by one artist (`name`, `limit`) |
| GET | `/api/tracks` | Token search over track and artist names (`q`, `limit`) |
| POST | `/api/recommend/new` | Fold-in for an unseen playlist. Body: `track_ids`, `k` (1-100), `exclude_artists` |
| GET | `/api/recommend/{playlist_id}` | Recommendations for a training playlist (hashed id), known tracks masked |
| GET | `/api/covers` | Deezer covers and preview URLs for up to 50 track ids (`ids`) |

Every returned track carries its placement tier (`learned`, `artist` or `genre`), which the
UI shows as a badge.

---

## Known Limitations

- Playlists sharing an owner are not independent samples; with the owner id discarded, that
  dependence cannot be controlled for in the split.
- Ranking metrics use 99 sampled negatives, which can disagree with full-catalogue ranking
  (Krichene & Rendle, 2020); the notebook reports both.
- The 5-core filter means the warm evaluation contains no cold items; cold-item quality is
  measured separately by proxy replacement.
- An attempt to retrain the item tower with item-ID dropout, so it could work from content
  alone, collapsed to a constant loss of ln 2 and was not pursued.
- The Playlist2vec snapshot is static; recent releases are only reachable through the artist
  and genre placement tiers.

---

## Use of Generative AI - Acknowledgement

In accordance with the University of Essex Online guidelines on academic integrity and the transparent use of generative AI tools in assessed work, the following declaration is made.

| Tool | Context of use |
|---|---|
| **Google Gemini 3.1 Pro** (Google DeepMind) | Supporting development agent during implementation: code scaffolding and boilerplate, debugging assistance, and documentation drafting and formatting. |
| **Claude Opus 5.5** (Anthropic), used through **GitHub Copilot** | Documentation improvement: auditing this README and the notebook's markdown against the notebook's actual outputs, correcting dataset provenance, and restructuring the documentation so it reflects the work carried out by the author. |

**Author responsibility:** All architectural decisions, design choices, modelling strategies, evaluation methodology, explainability interpretation, bias/ethics analysis, and critical evaluations presented in this submission are solely the intellectual work of the author. Every output produced by the AI tools was critically reviewed, validated, and where necessary revised or rejected by the author before incorporation. The author retains full responsibility for the correctness, originality, and academic integrity of the submitted work.

**Not AI-generated:** The task framing (playlist continuation over per-user rating prediction), dataset selection and justification, the 5-core filtering rationale, model design choices, evaluation protocol, explainability interpretation, and all critical ethical and deployment reflections documented in this submission are the author's original intellectual contributions.

---

## References

- James, G., Witten, D., Hastie, T. and Tibshirani, R. (2021). Resampling methods. In *An Introduction to Statistical Learning: with Applications in R*, 2nd edn, pp. 197-223. Springer. https://doi.org/10.1007/978-1-0716-1418-1_5
- Krichene, W. and Rendle, S. (2020). On sampled metrics for item recommendation. *KDD 2020*.
- Lundberg, S. M. and Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. and Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS 2013*.
- Pandya, M. (2023). *Spotify Tracks Dataset*. Kaggle. https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
- Papreja, P., Venkateswara, H. and Panchanathan, S. (2020). Representation, exploration and recommendation of playlists. In *Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2019)*, CCIS 1168, Springer. https://doi.org/10.1007/978-3-030-43887-6_50
- Papreja, P. (2020). *Playlist2vec: Spotify Million Playlist Dataset* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5002584
- Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *KDD 2016*.
