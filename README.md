# Spotify Playlist Continuation

**Module:** Comparative Intelligence and Ethics in Machine Learning Systems

**University of Essex Online**

A recommender system for **playlist continuation**: given a playlist's existing tracks, predict which further tracks a user is likely to add. Built on Track B (Recommender Systems) of the Unit 11 assignment, combining Spotify audio-feature metadata with real playlist-track co-occurrence data to compare a classical matrix-factorisation baseline against a deep two-tower neural model, with SHAP/LIME explainability, cold-start and sparsity diagnostics, and a discussion of fairness, privacy and deployment implications.

---

## Purpose

This is the practical development deliverable for the *Comparative Intelligence and Ethics in Machine Learning Systems* assignment (Unit 11, individual presentation, 40% of the module mark). It implements and compares two distinct modelling paradigms, classical ML and Deep Learning, on the same recommendation task, evaluated with ranking metrics appropriate to implicit-feedback recommender systems (RMSE as a baseline regression check, Recall@k and NDCG@k as the primary ranking metrics), interpreted through model-agnostic explainability techniques, and critically assessed for subgroup performance disparities and real-world ethical and deployment considerations.

---

## Task Framing

Public Spotify data has no persistent user identifier, so this project does **not** frame the task as per-user rating prediction. Instead, each **playlist** is treated as the unit of implicit preference: a track's presence in a playlist is treated as a positive implicit-feedback signal, and the task becomes predicting which unseen tracks belong in a given playlist (playlist continuation), evaluated in a leave-one-out ranking scheme against sampled negatives. Full rationale, citations and the accompanying ethics discussion are documented in the companion essay/presentation.

---

## Datasets

| Dataset | Source | Role |
|---|---|---|
| **Spotify Tracks Dataset** | Kaggle (maharshipandya/-spotify-tracks-dataset) | Per-track audio features (danceability, energy, tempo, key, valence, etc.) used as content-side signal |
| **Playlist2vec (Spotify Million Playlist Dataset dump)** | Zenodo (10.5281/zenodo.5002584) | Real playlist-to-track membership, used as the implicit collaborative-filtering signal |

The two datasets are joined on track identity, keeping only tracks that have both Kaggle audio features and at least one playlist membership in the Zenodo dump.

---

## Pipeline Overview

### 1. Data Preparation
- Kaggle audio-feature rows are aggregated from one row per `(track_id, genre)` into one row per track, with genres collapsed into a list.
- Invalid measurements are cleaned: rows with `tempo == 0` (silence/corrupt readings) are dropped; `time_signature == 0` is imputed with the modal value, since it is categorical rather than a genuine zero.
- The 10.7 GB Playlist2vec MySQL dump is streamed into a local SQLite database (`spotify.db`) rather than loaded into memory in full, as a one-off cost reused across later runs.
- Playlist-track interactions are restricted to tracks that already have Kaggle audio features, since a track with no features cannot be modelled regardless of playlist frequency.
- **5-core filtering** is applied iteratively (rather than uniform random sampling, which would break the collaborative signal) so that every retained playlist and track has at least 5 interactions, consistent with standard practice for mitigating power-law noise in sequential/collaborative recommendation research.

### 2. Sparsity Diagnostics
Matrix density, interactions-per-playlist and interactions-per-track distributions are computed and visualised to confirm the expected extreme sparsity of the interaction matrix, informing both model choice (matrix factorisation assumes sparsity) and the need for explicit cold-start handling in the evaluation split.

### 3. Modelling
An 80/20 playlist-level train/test split is used throughout.

| Model | Type | Description | Test RMSE |
|---|---|---|---|
| **TruncatedSVD** | Classical ML baseline | Matrix factorisation over the sparse playlist-track interaction matrix (50 latent components); RMSE computed on held-out interactions | **0.8879** |
| **Two-Tower Neural Network** | Deep Learning | Learned playlist and track ID embeddings fused with a dense audio-feature tower (with batch normalisation), trained to score playlist-track affinity | **0.0590** |

The two-tower model reduces RMSE by roughly 93% relative to the SVD baseline. Read in isolation, this signals that the non-linear fusion of ID embeddings and audio features gives the deep architecture substantially greater fitting capacity than the linear SVD factorisation.

> **Caveat - RMSE is a misleading metric for implicit feedback.** This RMSE was computed on a test set built from a 1-positive-to-1-negative ratio, and separating one real interaction from one random one is a comparatively easy binary discrimination task: it rewards the model for confidently pushing the two classes apart, not for ranking correctly against the full space of plausible alternatives. Distinguishing two songs is not the same task as recommending one. Under that same naive 1:1 setup, NDCG saturated at **1.0000**, a statistical mirage rather than evidence of real recommendation quality. RMSE is therefore reported here only to evidence the model's raw fitting capacity, not as a claim about recommendation performance. The Leave-One-Out evaluation below, with 99 negatives per positive, is the only metric in this project treated as valid evidence of real-world ranking performance.

### 4. Evaluation
Models are compared under a **leave-one-out (LOO)** ranking protocol: for each test playlist, one held-out positive track is ranked against 99 sampled negatives it never appeared with, and **Recall@k** and **NDCG@k** are reported at k = 5, 10, 20.

This design is a deliberate choice rather than a convenience, and it exists specifically to close the loophole exposed above. Ranking a positive against the *full* item catalogue would understate real recommender difficulty (with tens of thousands of tracks, almost any reasonable model scores near-perfect recall by pure chance), while ranking against too few negatives, as the 1:1 RMSE test set did, inflates Recall@k and NDCG@k artificially. Sampling 99 negatives per positive, following standard practice in implicit-feedback recommendation literature, forces the model to demonstrate whether the class-separation ability behind its low RMSE actually generalises into useful ranking under a realistic number of competing candidates, rather than simply confirming it can tell a real track from a random one.

| Metric | SVD Baseline | Two-Tower NN |
|---|---|---|
| Recall@5 | `X.XXX` | `X.XXX` |
| Recall@10 | `X.XXX` | `X.XXX` |
| Recall@20 | `X.XXX` | `X.XXX` |
| NDCG@5 | `X.XXX` | `X.XXX` |
| NDCG@10 | `X.XXX` | `X.XXX` |
| NDCG@20 | `X.XXX` | `X.XXX` |

*(Placeholders to be replaced with the values produced by the multi-k LOO evaluation cell once it finishes running.)*

### 5. Explainability
- **SHAP** (KernelExplainer, with a k-means-summarised background of 50 centroids) quantifies how each audio feature pushes the two-tower model's prediction for a given playlist-track pair.
- **LIME** (`LimeTabularExplainer`, regression mode) is applied to the same prediction as a second, independent local explanation, cross-checked against the SHAP result.

**Key finding:** cross-validating SHAP and LIME on the same playlist-track prediction, two methods with unrelated underlying mechanisms (game-theoretic attribution vs. local surrogate modelling), independently converged on the same dominant driver: `energy` consistently carries the largest attribution weight among the audio features. This agreement across two structurally different explainability techniques is itself evidence that the model's audio-feature tower is not behaving as an inscrutable black box: its predictions are traceable to a specific, musically interpretable signal rather than to noise or spurious correlations picked up during training.

### 6. Bias, Fairness and Ethics
Subgroup diagnostics (e.g. performance for low-activity/cold-start playlists vs. high-activity ones), data provenance and privacy considerations (no persistent user IDs in the public data, playlist-level rather than individual-level signal), and the real-world consequences of false positives/negatives in a recommendation context are discussed in the companion presentation, alongside a justified recommendation on which model is best suited to real-world deployment.

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
