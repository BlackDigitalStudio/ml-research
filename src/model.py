from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import joblib
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
        self._xgb_models: list[xgb.Booster] = []  # 3 XGBoost
        self._lgb_model = None  # LightGBM
        self._logreg_model = None  # LogisticRegression
        self._logreg_features: list[int] = []  # top-5 feature indices
        self._encoder_mtime: float = 0.0
        self._xgb_mtimes: list[float] = [0.0, 0.0, 0.0]
        self._lgb_mtime: float = 0.0
        self._logreg_mtime: float = 0.0
        self._logreg_feat_mtime: float = 0.0
        self._loaded = False
        self._last_uncertainty: float = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def n_models(self) -> int:
        count = len(self._xgb_models)
        if self._lgb_model is not None:
            count += 1
        if self._logreg_model is not None:
            count += 1
        return count

    @property
    def last_uncertainty(self) -> float:
        return self._last_uncertainty

    def load(self) -> bool:
        enc_path = self._model_dir / "encoder_latest.pt"

        if not enc_path.exists():
            logger.warning("Encoder file not found: %s", enc_path)
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

        loaded_count = 0

        # Load 3 XGBoost models
        self._xgb_models = []
        for i in range(3):
            xgb_path = self._model_dir / f"xgb_{i}_latest.json"
            if xgb_path.exists():
                try:
                    booster = xgb.Booster()
                    booster.load_model(str(xgb_path))
                    self._xgb_models.append(booster)
                    self._xgb_mtimes[i] = xgb_path.stat().st_mtime
                    logger.info("Loaded XGBoost %d from %s", i, xgb_path.name)
                    loaded_count += 1
                except Exception as e:
                    logger.error("Failed to load XGBoost %d: %s", i, e)
            else:
                logger.warning("XGBoost %d not found: %s", i, xgb_path)

        # Load LightGBM
        lgb_path = self._model_dir / "lgb_latest.txt"
        if lgb_path.exists():
            try:
                import lightgbm as lgb
                self._lgb_model = lgb.Booster(model_file=str(lgb_path))
                self._lgb_mtime = lgb_path.stat().st_mtime
                logger.info("Loaded LightGBM from %s", lgb_path.name)
                loaded_count += 1
            except Exception as e:
                logger.error("Failed to load LightGBM: %s", e)

        # Load LogisticRegression + feature indices
        logreg_path = self._model_dir / "logreg_latest.pkl"
        feat_path = self._model_dir / "logreg_features.json"
        if logreg_path.exists() and feat_path.exists():
            try:
                self._logreg_model = joblib.load(logreg_path)
                self._logreg_mtime = logreg_path.stat().st_mtime
                with open(feat_path) as f:
                    self._logreg_features = json.load(f)
                self._logreg_feat_mtime = feat_path.stat().st_mtime
                logger.info("Loaded LogisticRegression from %s (features: %s)",
                            logreg_path.name, self._logreg_features)
                loaded_count += 1
            except Exception as e:
                logger.error("Failed to load LogisticRegression: %s", e)

        # Need at least 3 models to be useful
        self._loaded = loaded_count >= 3
        logger.info("Ensemble: %d/%d models loaded, ready=%s",
                     loaded_count, 5, self._loaded)
        return self._loaded

    def check_reload(self) -> bool:
        enc_path = self._model_dir / "encoder_latest.pt"
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

        # Check 3 XGBoost models
        for i in range(3):
            xgb_path = self._model_dir / f"xgb_{i}_latest.json"
            mtime = self._xgb_mtimes[i] if i < len(self._xgb_mtimes) else 0.0
            if xgb_path.exists() and xgb_path.stat().st_mtime > mtime:
                try:
                    booster = xgb.Booster()
                    booster.load_model(str(xgb_path))
                    # Extend list if needed
                    while len(self._xgb_models) <= i:
                        self._xgb_models.append(None)
                    self._xgb_models[i] = booster
                    while len(self._xgb_mtimes) <= i:
                        self._xgb_mtimes.append(0.0)
                    self._xgb_mtimes[i] = xgb_path.stat().st_mtime
                    logger.info("Hot-swapped XGBoost %d", i)
                    reloaded = True
                except Exception as e:
                    logger.error("XGBoost %d hot-swap failed: %s", i, e)

        # Check LightGBM
        lgb_path = self._model_dir / "lgb_latest.txt"
        if lgb_path.exists() and lgb_path.stat().st_mtime > self._lgb_mtime:
            try:
                import lightgbm as lgb
                self._lgb_model = lgb.Booster(model_file=str(lgb_path))
                self._lgb_mtime = lgb_path.stat().st_mtime
                logger.info("Hot-swapped LightGBM")
                reloaded = True
            except Exception as e:
                logger.error("LightGBM hot-swap failed: %s", e)

        # Check LogisticRegression
        logreg_path = self._model_dir / "logreg_latest.pkl"
        feat_path = self._model_dir / "logreg_features.json"
        if logreg_path.exists() and logreg_path.stat().st_mtime > self._logreg_mtime:
            try:
                self._logreg_model = joblib.load(logreg_path)
                self._logreg_mtime = logreg_path.stat().st_mtime
                if feat_path.exists():
                    with open(feat_path) as f:
                        self._logreg_features = json.load(f)
                    self._logreg_feat_mtime = feat_path.stat().st_mtime
                logger.info("Hot-swapped LogisticRegression")
                reloaded = True
            except Exception as e:
                logger.error("LogisticRegression hot-swap failed: %s", e)

        if reloaded:
            self._loaded = self.n_models >= 3

        return reloaded

    def predict(
        self, lob_tensor: np.ndarray, hand_features: np.ndarray
    ) -> tuple[int, float]:
        if not self._loaded:
            return FLAT, 0.0

        # CNN embedding
        t0 = time.monotonic()
        x = torch.from_numpy(lob_tensor).unsqueeze(0)  # (1, 3, 20, 50)
        with torch.no_grad():
            embedding = self._encoder(x).numpy().flatten()  # (64,)
        cnn_ms = (time.monotonic() - t0) * 1000

        # Concatenate embedding + hand-crafted features
        combined = np.concatenate([embedding, hand_features]).reshape(1, -1)

        # Collect votes from all loaded models
        t1 = time.monotonic()
        votes: list[int] = []
        confidences: list[float] = []

        # XGBoost models
        for i, model in enumerate(self._xgb_models):
            if model is None:
                continue
            try:
                proba = model.predict(xgb.DMatrix(combined))
                if proba.ndim == 1:
                    proba = proba.reshape(1, -1)
                cls = int(np.argmax(proba[0]))
                conf = float(proba[0][cls])
                votes.append(cls)
                confidences.append(conf)
            except Exception as e:
                logger.warning("XGBoost %d predict failed: %s", i, e)

        # LightGBM
        if self._lgb_model is not None:
            try:
                proba = self._lgb_model.predict(combined)
                if proba.ndim == 1:
                    proba = proba.reshape(1, -1)
                cls = int(np.argmax(proba[0]))
                conf = float(proba[0][cls])
                votes.append(cls)
                confidences.append(conf)
            except Exception as e:
                logger.warning("LightGBM predict failed: %s", e)

        # LogisticRegression
        if self._logreg_model is not None and self._logreg_features:
            try:
                proba = self._logreg_model.predict_proba(
                    combined[:, self._logreg_features]
                )
                cls = int(np.argmax(proba[0]))
                conf = float(proba[0][cls])
                votes.append(cls)
                confidences.append(conf)
            except Exception as e:
                logger.warning("LogReg predict failed: %s", e)

        ensemble_ms = (time.monotonic() - t1) * 1000

        if not votes:
            self._last_uncertainty = 0.0
            return FLAT, 0.0

        # Uncertainty check
        uncertainty = max(confidences) - min(confidences)
        self._last_uncertainty = uncertainty
        if uncertainty > 0.20:
            logger.debug(
                "Predict: FLAT (uncertainty=%.3f > 0.20) CNN=%.1fms ENS=%.1fms",
                uncertainty, cnn_ms, ensemble_ms,
            )
            return FLAT, 0.0

        # Count votes per class, need >= 3/5 agreement for non-FLAT
        vote_counts = [0, 0, 0]  # UP, DOWN, FLAT
        for v in votes:
            vote_counts[v] += 1
        majority_class = int(np.argmax(vote_counts))
        majority_count = vote_counts[majority_class]

        if majority_class != FLAT and majority_count < 3:
            logger.debug(
                "Predict: FLAT (no 3/5 agreement: UP=%d DOWN=%d FLAT=%d) CNN=%.1fms ENS=%.1fms",
                vote_counts[UP], vote_counts[DOWN], vote_counts[FLAT],
                cnn_ms, ensemble_ms,
            )
            return FLAT, 0.0

        # Mean confidence of majority voters
        majority_confs = [
            confidences[i] for i, v in enumerate(votes) if v == majority_class
        ]
        mean_conf = float(np.mean(majority_confs))

        logger.debug(
            "Predict: %s (%.2f, %d/%d agree, unc=%.3f) CNN=%.1fms ENS=%.1fms",
            LABELS[majority_class], mean_conf, majority_count, len(votes),
            uncertainty, cnn_ms, ensemble_ms,
        )
        return majority_class, mean_conf
