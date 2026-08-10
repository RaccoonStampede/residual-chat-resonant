#!/usr/bin/env python3
"""
config.py — Configuration management for ResidualChat-Resonant.

Loads .env file, validates required settings, and provides defaults.
"""

import logging
import os
from pathlib import Path


def _load_dotenv(path: str = ".env"):
    """Minimal .env loader (no external dependency required)."""
    env_path = Path(path)
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


class Config:
    """Central configuration object for ResidualChat-Resonant."""

    def __init__(self, env_file: str = ".env"):
        _load_dotenv(env_file)
        self.FLASK_PORT: int = int(os.environ.get("FLASK_PORT", 5000))
        self.FLASK_DEBUG: bool = os.environ.get("FLASK_DEBUG", "False").lower() in (
            "1",
            "true",
            "yes",
        )
        self.STORAGE_PATH: str = os.environ.get("STORAGE_PATH", "./data")
        self.DATABASE_URL: str = os.environ.get(
            "DATABASE_URL",
            f"sqlite:///{self.STORAGE_PATH}/residual.db",
        )
        self.SESSION_TIMEOUT: int = int(os.environ.get("SESSION_TIMEOUT", 3600))
        self.MAX_SESSIONS: int = int(os.environ.get("MAX_SESSIONS", 100))
        self.ENABLE_PERSISTENCE: bool = os.environ.get(
            "ENABLE_PERSISTENCE", "True"
        ).lower() in ("1", "true", "yes")
        self.RATE_LIMIT: int = int(os.environ.get("RATE_LIMIT", 10))
        self.RATE_LIMIT_WINDOW: int = int(os.environ.get("RATE_LIMIT_WINDOW", 60))
        self.LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

        # Derived: SQLite file path from DATABASE_URL
        db_url = self.DATABASE_URL
        if db_url.startswith("sqlite:///"):
            self.DB_PATH = db_url[len("sqlite:///"):]
        else:
            self.DB_PATH = f"{self.STORAGE_PATH}/residual.db"

        self._validate()
        self._configure_logging()

    def _validate(self):
        if not (1 <= self.FLASK_PORT <= 65535):
            raise ValueError(f"FLASK_PORT must be 1–65535, got {self.FLASK_PORT}")
        if self.MAX_SESSIONS < 1:
            raise ValueError(f"MAX_SESSIONS must be >= 1, got {self.MAX_SESSIONS}")
        if self.SESSION_TIMEOUT < 0:
            raise ValueError(f"SESSION_TIMEOUT must be >= 0, got {self.SESSION_TIMEOUT}")

    def _configure_logging(self):
        log_dir = self.STORAGE_PATH
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "residual.log")
        level = getattr(logging, self.LOG_LEVEL, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file),
            ],
        )

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# Module-level singleton (lazy)
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
