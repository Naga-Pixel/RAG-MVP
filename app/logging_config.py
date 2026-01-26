"""
Logging configuration for b_rag.

Usage:
    from app.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Message")
"""
import logging
import sys
from app.logging_utils import request_id_ctx

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("-")
        return True


def setup_logging(level: str = "INFO") -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | rid=%(request_id)s | %(message)s",

        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


    for h in logging.getLogger().handlers:
        h.addFilter(RequestIdFilter())

    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)

