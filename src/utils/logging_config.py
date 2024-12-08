import sys
from pathlib import Path
from loguru import logger

def setup_logging(log_path: str = "logs/strategy.log", level: str = "INFO"):
    """Configure logging for the strategy"""
    # Create logs directory if it doesn't exist
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level
    )
    
    # Add file handler
    logger.add(
        log_path,
        rotation="1 day",
        retention="1 month",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level
    )