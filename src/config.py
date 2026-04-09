from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    # Binance credentials
    api_key: str = ""
    api_secret: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Trading
    symbol: str = "BTCUSDT"
    secondary_symbol: str = "ETHUSDT"  # for eth_leading_signal
    leverage: int = 20
    position_size_pct: int = 95
    # TP/SL as percentage of price (3:1 ratio)
    stop_loss_pct: float = 0.10    # 0.1% of BTC price
    take_profit_pct: float = 0.20  # 0.2% of BTC price (2:1 ratio)
    # Commission: different for win (maker+maker) and loss (maker+taker)
    commission_win_pct: float = 0.04   # 0.04% round-trip (both maker)
    commission_loss_pct: float = 0.07  # 0.07% round-trip (entry maker + SL taker)
    max_daily_loss_pct: float = 10.0
    max_consecutive_losses: int = 10
    order_timeout_sec: float = 2.0
    position_timeout_sec: float = 60.0
    confidence_threshold: float = 0.58
    retrain_interval_hours: int = 4
    cnn_retrain_interval_hours: int = 24

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("/home/scalper/scalper-bot/data"))
    model_dir: Path = field(default_factory=lambda: Path("/home/scalper/scalper-bot/models"))
    log_dir: Path = field(default_factory=lambda: Path("/home/scalper/scalper-bot/logs"))

    # Binance endpoints
    rest_base: str = "https://fapi.binance.com"
    ws_base: str = "wss://fstream.binance.com"

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError("BINANCE_API_KEY and BINANCE_API_SECRET are required")
        if not 1 <= self.leverage <= 125:
            raise ValueError(f"LEVERAGE must be 1-125, got {self.leverage}")
        if not 0 < self.stop_loss_pct < 10:
            raise ValueError(f"STOP_LOSS_PCT must be 0-10, got {self.stop_loss_pct}")
        if not 0 < self.take_profit_pct < 10:
            raise ValueError(f"TAKE_PROFIT_PCT must be 0-10, got {self.take_profit_pct}")
        if not 0 < self.max_daily_loss_pct <= 100:
            raise ValueError(f"MAX_DAILY_LOSS_PCT must be 0-100, got {self.max_daily_loss_pct}")

        for d in (self.data_dir, self.model_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def ws_depth_url(self) -> str:
        s = self.symbol.lower()
        return f"{self.ws_base}/ws/{s}@depth@100ms"

    @property
    def ws_aggtrade_url(self) -> str:
        s = self.symbol.lower()
        return f"{self.ws_base}/ws/{s}@aggTrade"

    @property
    def ws_markprice_url(self) -> str:
        s = self.symbol.lower()
        return f"{self.ws_base}/ws/{s}@markPrice@1s"

    @property
    def ws_secondary_depth_url(self) -> str:
        s = self.secondary_symbol.lower()
        return f"{self.ws_base}/ws/{s}@depth@100ms"

    @property
    def ws_secondary_aggtrade_url(self) -> str:
        s = self.secondary_symbol.lower()
        return f"{self.ws_base}/ws/{s}@aggTrade"

    def log_params(self) -> None:
        logger.info("Config loaded:")
        logger.info("  symbol=%s (secondary=%s)  leverage=x%d", self.symbol, self.secondary_symbol, self.leverage)
        logger.info("  SL=%.2f%%  TP=%.2f%%  ratio=%.0f:1", self.stop_loss_pct, self.take_profit_pct, self.take_profit_pct / self.stop_loss_pct)
        logger.info("  commission: win=%.2f%% loss=%.2f%%", self.commission_win_pct, self.commission_loss_pct)
        logger.info("  confidence=%.2f  position=%d%%", self.confidence_threshold, self.position_size_pct)
        logger.info("  data_dir=%s", self.data_dir)


def load_config(env_path: str | Path = "config.env") -> Config:
    env_path = Path(env_path)
    env: dict[str, str] = {}

    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()

    def _get(key: str, default: str = "") -> str:
        return env.get(key, os.environ.get(key, default))

    cfg = Config(
        api_key=_get("BINANCE_API_KEY"),
        api_secret=_get("BINANCE_API_SECRET"),
        telegram_bot_token=_get("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_get("TELEGRAM_CHAT_ID"),
        symbol=_get("SYMBOL", "BTCUSDT"),
        secondary_symbol=_get("SECONDARY_SYMBOL", "ETHUSDT"),
        leverage=int(_get("LEVERAGE", "20")),
        position_size_pct=int(_get("POSITION_SIZE_PCT", "95")),
        stop_loss_pct=float(_get("STOP_LOSS_PCT", "0.05")),
        take_profit_pct=float(_get("TAKE_PROFIT_PCT", "0.15")),
        commission_win_pct=float(_get("COMMISSION_WIN_PCT", "0.04")),
        commission_loss_pct=float(_get("COMMISSION_LOSS_PCT", "0.07")),
        max_daily_loss_pct=float(_get("MAX_DAILY_LOSS_PCT", "5")),
        max_consecutive_losses=int(_get("MAX_CONSECUTIVE_LOSSES", "10")),
        order_timeout_sec=float(_get("ORDER_TIMEOUT_SEC", "2")),
        position_timeout_sec=float(_get("POSITION_TIMEOUT_SEC", "60")),
        confidence_threshold=float(_get("CONFIDENCE_THRESHOLD", "0.58")),
        retrain_interval_hours=int(_get("RETRAIN_INTERVAL_HOURS", "4")),
        cnn_retrain_interval_hours=int(_get("CNN_RETRAIN_INTERVAL_HOURS", "24")),
        data_dir=Path(_get("DATA_DIR", "/home/scalper/scalper-bot/data")),
        model_dir=Path(_get("MODEL_DIR", "/home/scalper/scalper-bot/models")),
        log_dir=Path(_get("LOG_DIR", "/home/scalper/scalper-bot/logs")),
    )
    cfg.log_params()
    return cfg
