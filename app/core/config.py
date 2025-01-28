from pydantic_settings import BaseSettings
from typing import List
import os
from functools import lru_cache


class Settings(BaseSettings):

    # Log Analyzer Settings
    MODEL_NAME: str = "llama3.2"
    COLLECTION_NAME: str = "log_history"
    SIMILARITY_THRESHOLD: float = 0.92
    MAX_HISTORY_ENTRIES: int = 100
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001
    OLLAMA_HOST: str = "localhost"
    OLLAMA_PORT: int = 11434
    ANALYSIS_WINDOW: int = 20
    ALERT_THRESHOLD: float = 0.85
    RETENTION_DAYS: int = 30
    BATCH_SIZE: int = 100

    # Notification Settings
    ENABLE_NOTIFICATIONS: bool = False
    TELEGRAM_BOT_TOKEN: str = None
    TELEGRAM_CHAT_IDS: List[str] = []
    SLACK_WEBHOOK_URLS: List[str] = []
    SLACK_CHANNEL: str = "#monitoring"
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = None
    SMTP_PASSWORD: str = None
    # Notification Severity Levels
    NOTIFICATION_MIN_SEVERITY_TELEGRAM: str = "high"
    NOTIFICATION_MIN_SEVERITY_SLACK: str = "medium"
    NOTIFICATION_MIN_SEVERITY_EMAIL: str = "low"
    CACHE_TTL: int = 60
    EMAIL_RECIPIENTS: List[str] = []
    MIN_SEVERITY_TELEGRAM: str = "warning"
    MIN_SEVERITY_SLACK: str = "info"
    MIN_SEVERITY_EMAIL: str = "error"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance based on environment.
    Uses lru_cache to avoid reading the .env file on every call
    """
    environment = os.getenv("ENVIRONMENT", "dev")
    env_file = f"{environment}.env"

    # Check if environment-specific file exists, otherwise fall back to default .env
    if os.path.exists(env_file):
        return Settings(_env_file=env_file)
    return Settings()
