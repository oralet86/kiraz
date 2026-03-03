"""
Simple logging system for Cherry Projects.
"""

import logging
import sys
from paths import LOGS_DIR, UNIVERSAL_LOG_FILE


# Setup logging once
def setup_logging():
    """Setup simple logging to console and file."""
    LOGS_DIR.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("cherry_project")
    logger.setLevel(logging.INFO)

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(UNIVERSAL_LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent propagation
    logger.propagate = False

    return logger


# Initialize logger once
_logger = setup_logging()


def get_logger():
    """Get the logger instance."""
    return _logger


def add_log_file(file_path):
    """Add additional log file."""
    handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    _logger.addHandler(handler)


# Export logger
logger = get_logger()
