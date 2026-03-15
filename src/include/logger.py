import sys
from pathlib import Path

from loguru import logger


def setup_global_logger(output_dir: str | Path) -> None:
    """Configure the shared console and file logger used by the training pipeline."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "training.log"
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{message}</level>", level="INFO", colorize=True)
    logger.add(log_path, format="{time:MM-DD HH:mm:ss} {name}:{function}:{line} - {message}", level="INFO")
    logger.info(f"Global logger initialized. Logs are saving to: {log_path}")
