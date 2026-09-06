# Spotify Playlist Continuation

**Module:** Comparative Intelligence and Ethics in Machine Learning Systems

**University of Essex Online**

A recommender system for **playlist continuation**: given a playlist's existing tracks, predict which further tracks a user is likely to add. Built on Track B (Recommender Systems) of the Unit 11 assignment, combining Spotify audio-feature metadata with real playlist-track co-occurrence data to compare a classical matrix-factorisation baseline against a deep two-tower neural model, with SHAP/LIME explainability, cold-start and sparsity diagnostics, and a discussion of fairness, privacy and deployment implications.

---

## Purpose

This is the practical development deliverable for the *Comparative Intelligence and Ethics in Machine Learning Systems* assignment (Unit 11, individual presentation, 40% of the module mark). It implements and compares two distinct modelling paradigms, classical ML and Deep Learning, on the same recommendation task, evaluated with ranking metrics appropriate to implicit-feedback recommender systems (Recall@k and NDCG@k under a leave-one-out protocol), interpreted through model-agnostic explainability techniques, and critically assessed for subgroup performance disparities, long-tail exposure bias, and real-world ethical and deployment considerations.

---

## Task Framing

This project does **not** frame the task as per-user rating prediction. The dump *does* expose a persistent user identifier, the owner handle embedded in `playlist.uri`, so this is a deliberate design decision rather than a limitation of the data: that identifier is discarded at ingestion, and each **playlist** is treated as the unit of implicit preference. A track's presence in a playlist is a positive implicit-feedback signal, and the task becomes predicting which unseen tracks belong in a given playlist (playlist continuation), evaluated in a leave-one-out ranking scheme against sampled negatives.

Modelling per person would convert context-level inference into individual profiling on data whose subjects never consented to it, and the handles are frequently real names that allow re-identification against public Spotify profiles.

**Known limitation.** Because one owner may hold several playlists, playlists are not strictly independent samples. Discarding the owner id means that dependence cannot be controlled for in the train/test split.

> Rationale, trade-offs and the full critical/ethical discussion live in [docs/presentation-notes.md](docs/presentation-notes.md). This README documents the solution.

---

## Datasets

| Dataset | Source | Role |
|---|---|---|
| **Spotify Tracks Dataset** | Kaggle (maharshipandya/-spotify-tracks-dataset) | Per-track audio features (danceability, energy, tempo, key, valence, etc.) used as the content-side signal. |
| **Spotify Million Playlist Dataset (RecSys 2018)** via Playlist2vec dump | Zenodo (10.5281/zenodo.5002584) | Real playlist-to-track membership, used as the implicit collaborative-filtering signal. |

The two datasets are joined on track identity, keeping only tracks that possess both Kaggle audio features and at least one playlist membership in the MPD dump.

### Dataset Provenance & License Compliance

The Spotify Million Playlist Dataset (MPD) was released for the RecSys Challenge 2018 under a research-only licence and is consumed here via a secondary Zenodo mirror. Operating constraints applied to this project and its deployment:

* **Academic use only** - an academic Proof of Concept for university evaluation; not commercialised, monetised or offered to consumers.
* **Deployment isolation** - ephemeral, restricted to academic reviewers, not search-indexed, no advertising, no external user-data collection.
* **No data redistribution** - the backend never hosts or serves the raw MPD database; it returns inference results computed from exported model weights.

Provenance and consent implications are discussed in [docs/presentation-notes.md](docs/presentation-notes.md#data-provenance-and-consent).

### Data Minimization & PII Handling

Two columns of the `playlist` table are stripped inside the dump-to-SQLite loader ([mysql_to_sqlite.py](mysql_to_sqlite.py), `REDACTED_COLUMNS`). Both are removed from the generated DDL *and* from every `INSERT` tuple, so neither is ever written to local storage rather than merely left unqueried:

| Column | Why |
|---|---|
| `name` | User-generated free text leaking PII and special category data (emotional states, religious beliefs, sexual orientation) |
| `uri` | Embeds the owner's Spotify handle (`spotify:user:<id>:playlist:<id>`) in 100% of rows, frequently a real name rather than an opaque id, allowing re-identification against public profiles |

The loader logs each redacted column and aborts instead of falling back to positional inserts if a table's row width does not match its declared schema. Track, album and artist metadata is retained as public catalogue data.

The serialised training artefacts (`cache/dl_cache.pkl`) hold only anonymised integer mappings (`user_idx`, `item_idx`) plus numeric audio features, so the deployed API neither loads nor exposes text metadata.

This is data minimisation, not a claim of regulatory compliance: no lawful basis or consent exists for the underlying processing, which is why the system stays an academic PoC.

---

## Pipeline Overview

### 1. Data Preparation
- Kaggle audio-feature rows are aggregated from one row per `(track_id, genre)` into one row per track, with genres collapsed into a list.
- Invalid measurements are cleaned: rows with `tempo == 0` (silence/corrupt readings) are dropped; `time_signature == 0` is imputed with the modal value, since it is categorical rather than a genuine zero.
- The 10.7 GB Playlist2vec MySQL dump is streamed into a local SQLite database (`spotify.db`) rather than loaded into memory in full, as a one-off cost reused across later runs.
- `track_playlist1` holds **125.4M** interactions, so the audio-feature filter is executed **inside SQLite** (via a temporary catalogue table and a one-off index on `track_playlist1(track_id)`) and only surviving rows are materialised in pandas. Loading the raw table would require tens of gigabytes; the filtered result is **7,218,586** interactions (5.8%), around 0.8 GB.
- The binding constraint is the **intersection** of the two sources rather than the size of the dump: of 89,584 Kaggle tracks with audio features, only 31,403 appear in the dump at all, falling to **22,748** after 5-core filtering. That is the effective *training* catalogue, and all downstream metrics - including the 99 sampled negatives and coverage@10 - are measured against it. It is not the *serving* ceiling: 26,360 of the tracks the dump never saw are by an artist that is in the trained pool, so they can still be placed (see [Placing a track the model has never seen](#placing-a-track-the-model-has-never-seen)).
- **5-core filtering** is applied iteratively (rather than uniform random sampling, which would break the collaborative signal) so that every retained playlist and track has at least 5 interactions, consistent with standard practice for mitigating power-law noise in sequential/collaborative recommendation research.
- Playlists **discarded** by the 5-core filter are kept aside as a cold-start evaluation cohort, so the filter stays a *training* decision and does not silently remove the low-activity population the bias analysis needs to examine.

### 2. Sparsity Diagnostics
Matrix density, interactions-per-playlist and interactions-per-track distributions are computed and visualised to confirm the expected extreme sparsity of the interaction matrix, informing both model choice (matrix factorisation assumes sparsity) and the need for explicit cold-start handling in the evaluation split.

### 3. Modelling
A single 80/20 train/test split over **interactions** (playlist-track pairs) is computed once and reused by both models, so the comparison is like-for-like. Audio features are standardised with a `StandardScaler` fitted exclusively on the training positives; the same fitted scaler is reused for evaluation, explainability and the API export, so no train/inference distribution mismatch can arise.

| Model | Type | Description |
|---|---|---|
| **TruncatedSVD** | Classical ML baseline | Matrix factorisation over the sparse playlist-track interaction matrix (50 latent components) |
| **Two-Tower Neural Network** | Deep Learning | Learned playlist and track ID embeddings fused with a dense audio-feature tower (with batch normalisation), trained with BCE against uniformly sampled negatives regenerated each epoch |

> **RMSE is reported per model as a diagnostic only, never as a comparison.** The two figures measure different things on different data (positives-only with an unbounded output vs. a balanced set with a bounded sigmoid), and RMSE has no meaningful interpretation for implicit feedback in any case. Reasoning in [docs/presentation-notes.md](docs/presentation-notes.md#3-evaluation-and-validation).

#### Does the audio tower earn its complexity?

Two ablations, answering different questions:

| Measurement | Result |
|---|---|
| Neutralise the audio **input** of the trained model | −6.1% NDCG@10; shuffling it instead costs −19% |
| Retrain **without** the audio tower (equivalent normalisation retained) | 0.7734 warm / 0.6692 cold / 0.5054 coherence, i.e. marginally *better* than 0.7724 / 0.6630 / 0.4996 |

The trained model genuinely uses audio - corrupting the signal hurts far more than removing
it - but the information is **redundant**: an identical model denied audio recovers it from
the co-occurrence signal, and converges to the same place a couple of epochs later.

**Caveat that matters for deployment.** The 5-core filter guarantees every track has at
least five interactions, so the evaluation contains no genuinely cold *items*. Content
features exist to place a track nobody has added to a playlist yet, and that scenario is
absent by construction - so it was constructed and measured separately, below.

#### Placing a track the model has never seen

Artist vectors (the mean of an artist's item vectors) form musically coherent
neighbourhoods, while the same artists in acoustic space do not:

| Artist | Learned space | Acoustic space |
|---|---|---|
| blink-182 | Sum 41 (0.94), The Offspring (0.89), Green Day (0.83) | The Bombpops, The Muffs, Hoodoo Gurus |
| Nirvana | Smashing Pumpkins (0.92), Pearl Jam (0.88) | Accept, The Mission, New Model Army |
| Metallica | Iron Maiden (0.89), Slayer (0.82) | Pantera, Nada Surf (0.91), Arch Enemy |
| Bad Bunny | Ozuna (0.99), Maluma (0.99), J Balvin (0.97) | Ozuna, Wizkid, Dua Lipa, Snoop Dogg |

`Metallica ≈ Nada Surf` at 0.91 is the point: nine acoustic descriptors capture loudness and
tempo, not affinity. Artist identity does capture it, which makes it the usable content
feature for cold items. Replacing **every** candidate with a proxy computed without reading
that track's own vector:

| Placement strategy | NDCG@10 | Applies to |
|---|---|---|
| Learned vector (warm ceiling) | 0.7724 | 22,748 trained tracks |
| **Artist proxy** - mean of the artist's other tracks | **0.6554** | 88% of the catalogue |
| Genre centroid only | 0.4441 | the remainder |
| Global popularity (reference) | 0.3766 | - |

An unseen track reaches **85% of the warm ceiling from its artist alone**, with no
retraining: it is a fold-in on the item side, symmetric to the playlist fold-in. Weighting
artists by track count versus one vote each is immaterial (0.6554 vs 0.6546); blending in
genre makes it worse (0.6332).

This lifts the served catalogue from 22,748 to **81,738 tracks (3.6×)**, of which 58,990
have no interaction history at all. Trained tracks keep their original indices, so the user
vectors and interaction matrix stay aligned.

> **Negative result, recorded deliberately.** The principled alternative - retraining the
> item tower with random item-ID dropout so it learns to work from content alone -
> collapsed: loss pinned at exactly 0.6931 = ln 2 and warm NDCG@10 fell to 0.0000 by epoch
> 4. Zeroing the ID embedding in 30% of rows gives `item_fusion` a bimodal distribution its
> BatchNorm cannot normalise, and the terminal ReLU then kills the gradient permanently. A
> dying-ReLU collapse rather than evidence against the approach; it was not pursued because
> the artist proxy delivers the same capability at no training cost.

### 4. Evaluation
Models are compared under a **leave-one-out (LOO)** ranking protocol: one held-out positive track is ranked against 99 sampled negatives the playlist never contained, and **Recall@k** and **NDCG@k** are reported at k = 5, 10, 20. Both models are scored on the *identical* candidate set for each playlist.

Results are reported for two cohorts, because a single average would hide the effect of history length:

- **Warm** - playlists that survived the 5-core filter and have a learned embedding.
- **Cold** - playlists *discarded* by the 5-core filter (2-4 interactions), which have no embedding at all.

#### Why the cold cohort exists

The 5-core filter is defensible as a **training** decision, since a playlist with two tracks cannot support a stable learned embedding. Applying it to **evaluation** as well would be a methodological contradiction: it deletes precisely the low-activity population the bias analysis is required to examine. The cold cohort recovers those playlists, restricted to tracks that survived into the core so that item factors and embeddings exist to score against.

#### How a playlist with no embedding is scored

Neither model has a user vector for these playlists, so both are **folded in** from the playlist's seed tracks under an identical protocol:

| Model | Fold-in |
|---|---|
| TruncatedSVD | `svd.transform(row)`, which for an uncentred sparse row is `row @ components_.T` |
| Two-Tower | Mean of the **item-tower** vectors of the seed tracks |

The two-tower fold-in must average post-fusion item vectors rather than raw ID embeddings: the user embeddings were trained to dot against the fused output, so averaging in the wrong space produces arbitrary rankings. This is the same code path the API uses to serve an unseen playlist, which makes this cohort the only test coverage that path has.

One track is held out per playlist and ranked against 99 negatives, matching the warm protocol so the cohorts are directly comparable. With a single relevant item and binary gain, NDCG@k reduces analytically to `1 / log2(rank + 1)` when the held-out track lands inside the top k, identical to the `ndcg_score` used on the warm cohort.

**Reading the result:** if the warm-to-cold drop exceeds the SVD-to-DL difference, history length matters more than architecture, which is a more useful finding than a leaderboard position.

#### Results

All systems are scored on the **same** candidate sets over 20,000 held-out playlists. Two
non-learned baselines set the floor (see section 6): global popularity, and popularity
weighted by how well a track's genres match the playlist's genre profile.

| System | R@5 | R@10 | R@20 | NDCG@5 | NDCG@10 | NDCG@20 |
|---|---|---|---|---|---|---|
| Global popularity | 0.4652 | 0.6472 | 0.8250 | 0.3178 | 0.3766 | 0.4216 |
| Genre popularity | 0.7925 | 0.8359 | 0.8438 | 0.6510 | 0.6655 | 0.6675 |
| TruncatedSVD | 0.8087 | 0.8818 | 0.9383 | 0.6878 | 0.7116 | 0.7260 |
| Two-Tower v1 | 0.7954 | 0.8906 | 0.9485 | 0.6271 | 0.6583 | 0.6730 |
| **Two-Tower v2** | - | **0.9379** | - | - | **0.7724** | - |

**The headline finding is v1, not v2.** In its first configuration the deep model was beaten
on NDCG by the classical baseline *and* by a two-line heuristic with no parameters. The
cause was the training task, not the architecture: with uniformly sampled negatives, global
popularity separated the classes on its own, so the model was never required to learn
context. Retraining with popularity-matched negatives and early stopping (section 10) moved
it from 0.658 to 0.772 and to first place.

#### Cold cohort (fold-in)

| Model | Warm NDCG@10 | Cold NDCG@10 | Drop |
|---|---|---|---|
| TruncatedSVD | 0.7116 | 0.5798 | 0.1318 |
| Two-Tower v1 | 0.6583 | 0.5341 | 0.1242 |
| **Two-Tower v2** | **0.7724** | **0.6630** | **0.1094** |

The warm-to-cold drop is larger than the gap between models, so **history length matters
more than architecture**. v2 is nonetheless the fairest of the three across cohorts as well
as the strongest overall, which is what supports the deployment recommendation.

> Metrics computed against sampled negatives are biased estimators of full-catalogue ranking (Krichene & Rendle, KDD 2020). See [docs/presentation-notes.md](docs/presentation-notes.md#3-evaluation-and-validation).

### 5. Explainability
Applied to the **final** model, with three complementary views:

| Technique | Scope | Faithfulness |
|---|---|---|
| **SHAP** (KernelExplainer, 50 k-means background centroids, 100 interactions) | Audio tower, IDs held fixed | Approximate, model-agnostic |
| **LIME** (`LimeTabularExplainer`, classification mode) | Same interaction, independent mechanism | Approximate, local surrogate |
| **Latent factor attribution** | TruncatedSVD | **Exact** - the score decomposes additively over factors |

**Findings.** Globally, `danceability` (0.041), `acousticness` (0.038) and
`instrumentalness` (0.035) carry the largest mean |SHAP|; `tempo` and `liveness` are close
to irrelevant. Total attribution magnitude is ~0.22 per prediction, so the audio tower
moves the probability meaningfully but does not dominate it. On a single interaction SHAP
and LIME agree in sign on 8 of 9 features (89%), which is evidence the local explanation is
stable rather than an artefact of either method - but note that the locally dominant feature
(`energy`) is only fourth globally, so local explanations must not be generalised.

**The comparative point.** SHAP and LIME can only reach the audio tower, so they explain
the slice of the prediction the audio features control, not the collaborative signal that
dominates it. The SVD score, being a dot product, decomposes *exactly* into per-factor
contributions with no approximation, and each factor can be named by the genres its item
loadings concentrate on. This inverts the usual assumption that the classical model is the
opaque one: here it is the only one with an exact explanation.

Interpretation and its limits: [docs/presentation-notes.md](docs/presentation-notes.md#4-explainability-findings).

### 6. Bias, Fairness and Ethics
The notebook produces four diagnostics; the reasoning behind each is in [docs/presentation-notes.md](docs/presentation-notes.md#5-ethics-and-bias).

| Diagnostic | What it measures |
|---|---|
| Subgroup NDCG@10 | Ranking quality split by playlist size, under the shared LOO protocol |
| Cold-start cohort | Fold-in performance for playlists excluded by the 5-core filter |
| Exposure metrics | Catalogue coverage@10, Gini of exposure, mean popularity percentile of recommended tracks |
| Lorenz curve | Full exposure distribution across the catalogue |
| Genre coherence@10 | Whether the top-10 matches the playlist's genre profile, under exact, behavioural and acoustic notions of genre similarity |

#### Measured exposure (v1 models, 2,000 playlists scored against the full catalogue)

| | SVD | Two-Tower v1 |
|---|---|---|
| Coverage@10 | 0.037 | 0.039 |
| Gini of exposure | 0.983 | 0.991 |
| Mean popularity percentile | 98.5 | 97.5 |

Against a catalogue baseline of 50, both concentrate almost all exposure on the top 2% of
tracks: roughly 840 of 22,748 tracks are ever recommended.

#### Genre coherence@10

Exact label matching is too strict, since recommending punk-rock to a pop-rock playlist is a
hit rather than a miss. Two similarity matrices soften it, and separating them is itself
diagnostic: **behavioural** similarity comes from genre co-occurrence within playlists,
**acoustic** similarity from genre centroids in audio-feature space.

| System | Exact | Behavioural | Acoustic |
|---|---|---|---|
| Genre popularity | 0.223 | 0.548 | 0.559 |
| TruncatedSVD | 0.195 | 0.486 | 0.551 |
| Two-Tower v1 | 0.097 | 0.337 | 0.485 |
| **Two-Tower v2** | - | **0.500** | - |
| Global popularity | 0.026 | 0.257 | 0.239 |
| Random | 0.010 | 0.094 | 0.214 |

Genre popularity leads the coherence metric *by construction*, since it ranks by exactly
that quantity; the meaningful comparisons are between the learned systems. Two incidental
findings are worth stating: the behavioural matrix confirms that `pop / rock / punk /
alternative` form a tight cluster while `jazz` and `classical` are isolated, and the
acoustic matrix shows audio features are poor genre discriminators (`jazz` is 0.98 similar
to `disney`, `classical` 0.99 to `ambient`), which is a concrete argument for why the
collaborative signal cannot be replaced by content alone.

#### Exposure bias: why the accuracy metrics are blind to it

RMSE, Recall@k and NDCG@k all answer the same question: *given a track the playlist actually contains, does the model rank it highly?* None can detect the failure mode below, because they only ever look at items the model was asked about.

**Where the bias comes from.** Playlist membership follows a power law, and two mechanisms amplify the skew:

1. Popular tracks generate proportionally more positive training examples, so their ID embeddings receive far more gradient updates than long-tail ones.
2. Negatives are sampled uniformly from the catalogue, so a random negative is almost always an obscure track. The model can reduce its loss simply by learning *"popular = positive, obscure = negative"*, without learning anything about the playlist it is meant to be completing.

**Why the metrics cannot see it.** The held-out positive is itself drawn from the same power-law distribution, so it is usually a popular track too. A model that ranks purely by popularity scores well while only ever recommending the head of the catalogue.

**How it is measured.** Both models score the **full catalogue** for a sample of playlists, with tracks already in the playlist masked out, and the resulting top-10 lists are pooled into:

- **Catalogue coverage@10** - fraction of distinct tracks that ever appear in any recommendation. Low coverage means most of the catalogue is unreachable.
- **Gini coefficient of exposure** - `0.0` = every track recommended equally often, `1.0` = all recommendations concentrate on a single track.
- **Mean popularity percentile** of recommended tracks, against a catalogue baseline of 50.
- **Lorenz curve** - the full exposure distribution rather than just its summary statistics.

**Why it matters ethically.** The cost is not paid by the listener, who simply sees mainstream suggestions, but by long-tail artists who are never surfaced. That omission is invisible: nobody complains about a recommendation they did not receive, so it generates no signal and no metric. Because recommendations shape listening and listening becomes the next training set, the concentration compounds with each retraining cycle.

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

The API deliberately has no deep-learning dependency: the item tower is applied once in the
notebook and only its output is shipped. See [SERVING.md](SERVING.md).

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
- A Kaggle account with API credentials configured (`~/.kaggle/kaggle.json`), for automatic dataset download
- Sufficient local disk space (~11 GB+) for the Playlist2vec MySQL dump and its SQLite conversion

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Create a `.env` file with any required credentials (e.g. Kaggle API token location) and run:
```bash
cp .env.example .env
# edit .env as needed
```

### 3. Run the notebook
```bash
jupyter notebook main.ipynb
```
On first run, the notebook will automatically download the Kaggle Spotify Tracks Dataset and the Zenodo Playlist2vec dump, and build the local SQLite database. These artefacts are cached locally so subsequent runs skip the download/conversion step.

### 4. Run the demo

The export cells at the end of the notebook write a `deploy/` bundle in which the item tower
has already been applied to the whole catalogue, so the API needs neither PyTorch nor the
training cache: each request is a fold-in plus one matrix-vector product.

```bash
# API on http://127.0.0.1:8000
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000

# UI on http://localhost:5173 (proxies /api to the API)
cd "frontend/Music Recommender" && npm install && npm run dev
```

Browse artists by genre and popularity, or search the catalogue directly, assemble a playlist
that does not exist in the training data, and the model folds it in from the seed tracks.
Every track carries a badge stating how it was placed - `trained`, `unseen · via artist` or
`unseen · via genre` - so a recommendation can be interrogated rather than taken on trust.

A playlist built **only from tracks the model never saw** (blink-182's *EDGING*, *Darkside*
and *I Really Wish I Hated You*, all released after the dump was collected) returns MxPx,
Pennywise, The Swellers, Millencolin and Masked Intruder: the punk neighbourhood, reached
without a single observed interaction for any seed.

**[SERVING.md](SERVING.md) documents the serving layer in full** - the deployment bundle and
its integrity invariants, the placement tiers, the complete API reference, the interface, and
the limitations.

---

## Use of Generative AI - Acknowledgement

In accordance with the University of Essex Online guidelines on academic integrity and the transparent use of generative AI tools in assessed work, the following declaration is made:

**Tool:** Google Gemini 3.1 Pro (Google DeepMind)

**Context of use:** Google Gemini 3.1 Pro was used as a supporting development agent during the implementation phase of this project. Its role was limited to code generation assistance, debugging support, and drafting documentation.

**Author responsibility:** All architectural decisions, design choices, modelling strategies, evaluation methodology, explainability interpretation, bias/ethics analysis, and critical evaluations presented in this submission are solely the intellectual work of the author. Every output produced by the AI tool was critically reviewed, validated, and where necessary revised or rejected by the author before incorporation. The author retains full responsibility for the correctness, originality, and academic integrity of the submitted work.

**Scope of AI assistance:**
- Code scaffolding and boilerplate generation
- Debugging assistance
- Documentation drafting and formatting

**Not AI-generated:** The task framing (playlist continuation over per-user rating prediction), dataset selection and justification, the 5-core filtering rationale, model design choices, evaluation protocol, explainability interpretation, and all critical ethical and deployment reflections documented in this submission are the author's original intellectual contributions.

---

## References

TODO
