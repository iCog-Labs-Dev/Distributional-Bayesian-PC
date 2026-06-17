"""Project-wide logger: stdout + dbpcn.log at repo root."""
import logging
from pathlib import Path

_LOG_FILE = Path(__file__).parent / "dbpcn.log"
_FORMAT = "%(asctime)s [%(name)s] %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return a logger tagged with `name` (e.g. 'mnist', 'ood_eval').

    Emits to stdout and to dbpcn.log at the repo root.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter(_FORMAT, _DATEFMT)
        for h in (logging.StreamHandler(), logging.FileHandler(_LOG_FILE)):
            h.setFormatter(fmt)
            logger.addHandler(h)
        logger.propagate = False
    return logger
