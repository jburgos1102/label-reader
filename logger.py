import logging
import os


LOGGER_NAME = "label_reader"
DEFAULT_LOG_LEVEL = "INFO"


def _get_log_level():
    level_name = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
    return getattr(logging, level_name, logging.INFO)


log = logging.getLogger(LOGGER_NAME)
log.setLevel(_get_log_level())

if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    log.addHandler(handler)

log.propagate = False
