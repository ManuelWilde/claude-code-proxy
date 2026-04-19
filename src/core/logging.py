import logging
import os

# Read log level directly from env (avoid circular import with config module)
log_level = os.environ.get("LOG_LEVEL", "INFO").split()[0].upper()

valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
if log_level not in valid_levels:
    log_level = 'INFO'

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
for uvicorn_logger in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
    logging.getLogger(uvicorn_logger).setLevel(logging.WARNING)
