import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with HH:MM:SS timestamp and fixed-width level name.

    Safe to call multiple times; only sets up handlers once.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Already configured
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)5s] %(message)s",
        datefmt="%H:%M:%S",
    )


