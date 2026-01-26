import logging
import os
import sys
import uuid
from contextvars import ContextVar

# Context variable storing request_id for the current request
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Inject request_id into all log records
        record.request_id = request_id_ctx.get("-")
        return True

def setup_logging() -> None:
    """
    Minimal, production-safe logging config:
    - stdout handler (Docker captures it)
    - consistent format including request_id
    - no fancy JSON yet (can upgrade later)
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    root = logging.getLogger()
    root.setLevel(log_level)

    # Clear existing handlers to avoid duplicate logs
    root.handlers = []

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    fmt = (
        "%(asctime)s | %(levelname)s | %(name)s | rid=%(request_id)s | %(message)s"
    )
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(RequestIdFilter())

    root.addHandler(handler)

def new_request_id() -> str:
    return uuid.uuid4().hex[:12]
