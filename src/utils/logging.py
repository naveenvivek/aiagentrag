"""
Logging utilities for the AI Agent RAG application.
"""
import logging
import os
from pathlib import Path
from typing import Optional

from src.utils.config import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
    
    Returns:
        Configured logger instance
    """
    level = log_level or settings.log_level
    file_path = log_file or settings.log_file
    
    # Create logs directory if it doesn't exist
    log_dir = Path(file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(file_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("aiagentrag")
    logger.info(f"Logging initialized. Level: {level}, File: {file_path}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(f"aiagentrag.{name}")
