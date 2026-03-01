"""
Centralized Logging System for Cherry Projects

Provides unified logging with:
1. Console output
2. Universal logs.log file  
3. Date-specific log files in logs/ directory
4. Easy import: from log import logger

Usage:
    from log import logger
    logger.info("Something!")
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from paths import LOGS_DIR, UNIVERSAL_LOG_FILE


# Global configuration
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class CentralizedLogger:
    """Centralized logging system with multiple output destinations."""
    
    def __init__(self):
        self._logger = None
        self._setup_complete = False
    
    def _setup_logging(self, name: str = "cherry_project") -> logging.Logger:
        """Set up the centralized logging system."""
        if self._setup_complete and self._logger is not None:
            return self._logger
        
        # Create logs directory if it doesn't exist
        LOGS_DIR.mkdir(exist_ok=True)
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(LOG_LEVEL)
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        
        # 1. Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 2. Universal Log File Handler (append mode)
        universal_handler = logging.FileHandler(UNIVERSAL_LOG_FILE, mode='a', encoding='utf-8')
        universal_handler.setLevel(LOG_LEVEL)
        universal_handler.setFormatter(formatter)
        logger.addHandler(universal_handler)
        
        # 3. Date-Specific Log File Handler
        date_str = datetime.now().strftime("%Y-%m-%d")
        date_log_file = LOGS_DIR / f"{date_str}.log"
        
        date_handler = logging.FileHandler(date_log_file, mode='a', encoding='utf-8')
        date_handler.setLevel(LOG_LEVEL)
        date_handler.setFormatter(formatter)
        logger.addHandler(date_handler)
        
        # Prevent propagation to root logger to avoid duplicate messages
        logger.propagate = False
        
        self._logger = logger
        self._setup_complete = True
        
        # Log the setup completion
        logger.info(f"Centralized logging initialized - Console, Universal: {UNIVERSAL_LOG_FILE}, Date: {date_log_file}")
        
        return logger
    
    def get_logger(self, name: str = "cherry_project") -> logging.Logger:
        """
        Get the centralized logger instance.
        
        Args:
            name: Logger name (optional, defaults to "cherry_project")
        
        Returns:
            Configured logger instance
        """
        return self._setup_logging(name)
    
    def set_level(self, level: int) -> None:
        """
        Change logging level for all handlers.
        
        Args:
            level: Logging level (logging.DEBUG, logging.INFO, etc.)
        """
        if not self._logger:
            self.get_logger()
        self._logger.setLevel(level)
        for handler in self._logger.handlers:
            handler.setLevel(level)
    
    def add_file_handler(self, file_path: Path, level: Optional[int] = None) -> None:
        """
        Add an additional file handler.
        
        Args:
            file_path: Path to additional log file
            level: Logging level for this handler (optional)
        """
        if not self._logger:
            self.get_logger()
        
        resolved = Path(file_path).resolve()
        for h in self._logger.handlers:
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == resolved:
                return
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        handler.setLevel(level or LOG_LEVEL)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        
        self._logger.addHandler(handler)
        self._logger.info(f"Added additional log handler: {file_path}")


# Global instance
_centralized_logger = CentralizedLogger()

# Public interface - lazy initialization to prevent repeated setup messages
_logger_instance = None

def get_logger() -> logging.Logger:
    """Get the centralized logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = _centralized_logger.get_logger()
    return _logger_instance

# Create a simple logger proxy for backward compatibility
class _LoggerProxy:
    """Simple proxy for lazy logger access."""
    def __getattr__(self, name):
        return getattr(get_logger(), name)

logger = _LoggerProxy()

# Additional utility functions
def set_log_level(level: int) -> None:
    """Set logging level for all handlers."""
    _centralized_logger.set_level(level)

def add_log_file(file_path: Path, level: Optional[int] = None) -> None:
    """Add an additional log file handler."""
    _centralized_logger.add_file_handler(file_path, level)


if __name__ == "__main__":
    # Test the logging system
    print("Testing centralized logging system...")
    
    # Test using the logger directly
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    print("Log files created in: {}".format(LOGS_DIR))
    print("Check these files:")
    print("  - Console output (above)")
    print(f"  - {UNIVERSAL_LOG_FILE}")
    print(f"  - {LOGS_DIR / datetime.now().strftime('%Y-%m-%d')}.log")
