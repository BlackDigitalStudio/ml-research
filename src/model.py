from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb

from src.config import Config

logger = logging.getLogger(__name__)

UP, DOWN, FLAT = 0, 1, 2
LABELS = {UP: "UP", DOWN: "DOWN", FLAT: "FLAT"}


class LOBEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout2d(0.2)
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Linear(64, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x  # (batch, 64)


class HybridModel:
    def __init__(self, config: Config) -> None:
        self._model_dir = config.model_dir
        self._encoder = LOBEncoder()
        self._encoder.eval()
        self._xgb: xgb.Booster | None = None
        self._encoder_mtime: float = 0.0
        self._xgb_mtime: float = 0.0
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> bool:
        enc_path = self._model_dir / "encoder_latest.pt"
        xgb_path = self._model_dir / "xgb_latest.json"

        if not enc_path.exists() or not xgb_path.exists():
            logger.warning("Model files not found: %s, %s", enc_path, xgb_path)
            return False

        try:
            state = torch.load(enc_path, map_location="cpu", weights_only=True)
            self._encoder.load_state_dict(state)
            self._encoder.eval()
            self._encoder_mtime = enc_path.stat().st_mtime
            logger.info("Loaded CNN encoder from %s", enc_path.name)
        except Exception as e:
            logger.error("Failed to load CNN encoder: %s", e)
            return False

        try:
            self._xgb = xgb.Booster()
            self._xgb.load_model(str(xgb_path))
            self._xgb_mtime = xgb_path.stat().st_mtime
            logger.info("Loaded XGBoost from %s", xgb_path.name)
        except Exception as e:
            logger.error("Failed to load XGBoost: %s", e)
            return False

        self._loaded = True
        return True

    def check_reload(self) -> bool:
        enc_path = self._model_dir / "encoder_latest.pt"
        xgb_path = self._model_dir / "xgb_latest.json"

        reloaded = False

        if enc_path.exists() and enc_path.stat().st_mtime > self._encoder_mtime:
            try:
                state = torch.load(enc_path, map_location="cpu", weights_only=True)
                self._encoder.load_state_dict(state)
                self._encoder.eval()
                self._encoder_mtime = enc_path.stat().st_mtime
                logger.info("Hot-swapped CNN encoder")
                reloaded = True
            except Exception as e:
                logger.error("CNN hot-swap failed: %s", e)

        if xgb_path.exists() and xgb_path.stat().st_mtime > self._xgb_mtime:
            try:
                booster = xgb.Booster()
                booster.load_model(str(xgb_path))
                self._xgb = booster
                self._xgb_mtime = xgb_path.stat().st_mtime
                logger.info("Hot-swapped XGBoost model")
                reloaded = True
            except Exception as e:
                logger.error("XGBoost hot-swap failed: %s", e)

        return reloaded

    def predict(
        self, lob_tensor: np.ndarray, hand_features: np.ndarray
    ) -> tuple[int, float]:
        if not self._loaded or self._xgb is None:
            return FLAT, 0.0

        # CNN embedding
        t0 = time.monotonic()
        x = torch.from_numpy(lob_tensor).unsqueeze(0)  # (1, 3, 20, 50)
        with torch.no_grad():
            embedding = self._encoder(x).numpy().flatten()  # (64,)
        cnn_ms = (time.monotonic() - t0) * 1000

        # Concatenate embedding + hand-crafted features
        combined = np.concatenate([embedding, hand_features]).reshape(1, -1)

        # XGBoost prediction
        t1 = time.monotonic()
        dmatrix = xgb.DMatrix(combined)
        proba = self._xgb.predict(dmatrix)  # (1, 3)

        if proba.ndim == 1:
            # binary or single-row output
            proba = proba.reshape(1, -1)

        xgb_ms = (time.monotonic() - t1) * 1000

        predicted_class = int(np.argmax(proba[0]))
        confidence = float(proba[0][predicted_class])

        logger.debug(
            "Predict: %s (%.2f) CNN=%.1fms XGB=%.1fms",
            LABELS[predicted_class], confidence, cnn_ms, xgb_ms,
        )
        return predicted_class, confidence
