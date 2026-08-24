import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager


# --- Pydantic Models ---
class NewPlaylistRequest(BaseModel):
    track_ids: list[str]


# --- PyTorch Model Definition ---
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, num_audio_features, embed_dim=32):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim)
        self.audio_layer = nn.Sequential(
            nn.Linear(num_audio_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
        )
        self.item_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()
        )

    def forward(self, user_idx, item_idx, audio_features):
        u_emb = self.user_embedding(user_idx)
        i_emb = self.item_embedding(item_idx)
        a_emb = self.audio_layer(audio_features)
        i_combined = self.item_fusion(torch.cat([i_emb, a_emb], dim=1))
        dot_product = (u_emb * i_combined).sum(dim=1)
        return torch.sigmoid(dot_product)


# Global State para mantener los tensores en RAM
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar caché y diccionarios
    with open("../cache/dl_cache.pkl", "rb") as f:
        cache = pickle.load(f)

    state["playlist_map"] = cache["playlist_map"]
    state["track_map"] = cache["track_map"]
    state["reverse_track_map"] = {v: k for k, v in cache["track_map"].items()}

    # Construir historial de usuarios para evitar recomendar lo ya conocido (Filtro Anti-Data Leakage)
    train_df = cache["train_dl_df"]
    positives = train_df[train_df["rating"] == 1.0]
    state["user_history"] = (
        positives.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )

    # Cargar el modelo
    num_users = len(state["playlist_map"])
    num_items = len(state["track_map"])
    num_audio_features = len(cache["audio_features"])

    model = TwoTowerModel(num_users, num_items, num_audio_features)
    model.load_state_dict(
        torch.load("../models/two_tower_model.pth", weights_only=True)
    )
    model.eval()
    state["model"] = model

    # Cargar tensores de catálogo para inferencia rápida
    state["audio_tensors"] = torch.load("../cache/audio_tensors.pt", weights_only=True)
    state["all_items_tensor"] = torch.arange(num_items, dtype=torch.long)

    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/recommend/{playlist_id}")
def recommend_historical(playlist_id: str, k: int = 10):
    if playlist_id not in state["playlist_map"]:
        raise HTTPException(
            status_code=404, detail="Playlist no encontrada en el dataset histórico."
        )

    u_idx = state["playlist_map"][playlist_id]

    # Inferencia Vectorizada
    with torch.no_grad():
        u_tensor = torch.full(
            (len(state["all_items_tensor"]),), u_idx, dtype=torch.long
        )
        preds = state["model"](
            u_tensor, state["all_items_tensor"], state["audio_tensors"]
        ).numpy()

    known_items = state["user_history"].get(u_idx, set())
    top_indices = np.argsort(preds)[::-1]

    recommendations = []
    for idx in top_indices:
        if idx not in known_items:
            recommendations.append(state["reverse_track_map"][idx])
        if len(recommendations) == k:
            break

    return {"playlist_id": playlist_id, "recommendations": recommendations}


@app.post("/recommend/new")
def recommend_new(request: NewPlaylistRequest, k: int = 10):
    track_ids = request.track_ids
    valid_indices = [
        state["track_map"][t] for t in track_ids if t in state["track_map"]
    ]

    if not valid_indices:
        raise HTTPException(
            status_code=400, detail="Ninguna canción enviada existe en el catálogo."
        )

    t_tensor = torch.tensor(valid_indices, dtype=torch.long)

    with torch.no_grad():
        # Fallback: Mean Pooling de las canciones elegidas para fabricar una firma de usuario
        item_embs = state["model"].item_embedding(t_tensor)
        synthetic_user_emb = item_embs.mean(dim=0, keepdim=True)

        # Cruzar contra todo el catálogo
        all_i_embs = state["model"].item_embedding(state["all_items_tensor"])
        all_a_embs = state["model"].audio_layer(state["audio_tensors"])
        all_i_combined = state["model"].item_fusion(
            torch.cat([all_i_embs, all_a_embs], dim=1)
        )

        dot_product = (synthetic_user_emb * all_i_combined).sum(dim=1)
        preds = torch.sigmoid(dot_product).numpy()

    known_items = set(valid_indices)
    top_indices = np.argsort(preds)[::-1]

    recommendations = []
    for idx in top_indices:
        if idx not in known_items:
            recommendations.append(state["reverse_track_map"][idx])
        if len(recommendations) == k:
            break

    return {"recommendations": recommendations}
