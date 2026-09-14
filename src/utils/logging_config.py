import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/agent.log"),
        ],
    )


def get_env(key: str, default: str = "") -> str:
    import os
    return os.environ.get(key, default)