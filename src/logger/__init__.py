import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "app.log")
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3
ENABLE_FILE_LOGGING = os.getenv("ENABLE_FILE_LOGGING", "false").lower() in ("true", "1", "yes")

_configured = False


def configure_logging(level: int = None) -> logging.Logger:
    """
    Configures the root logger with a console handler (stdout) and optionally a rotating file handler.
    Thread-safe and idempotent: will not add duplicate handlers.
    """
    global _configured
    root_logger = logging.getLogger()

    log_level_env = os.getenv("LOG_LEVEL", "INFO").upper()
    resolved_level = level if level is not None else getattr(logging, log_level_env, logging.INFO)
    root_logger.setLevel(resolved_level)

    if _configured or root_logger.handlers:
        return root_logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler (standard for containers and cloud logging)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(resolved_level)
    root_logger.addHandler(console_handler)

    # Optional rotating file handler
    if ENABLE_FILE_LOGGING:
        try:
            repo_root = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
            log_dir_path = os.path.join(repo_root, LOG_DIR)
            os.makedirs(log_dir_path, exist_ok=True)
            log_file_path = os.path.join(log_dir_path, LOG_FILE_NAME)

            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=MAX_LOG_SIZE,
                backupCount=BACKUP_COUNT,
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(resolved_level)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning("Failed to configure file logging: %s", e)

    _configured = True
    return root_logger


def get_logger(name: str = "sentiment_analysis") -> logging.Logger:
    """Get a logger instance with configured handlers."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)


# Exported standard logger instance
logger = configure_logging()