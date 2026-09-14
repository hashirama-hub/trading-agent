import time
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class HealthCheck:
    def __init__(self):
        self.last_check = time.time()
        self.errors: list = []

    def check(self) -> Dict:
        now = time.time()
        uptime = now - self.last_check

        status = {
            "healthy": True,
            "uptime_seconds": uptime,
            "errors_last_hour": len([e for e in self.errors if time.time() - e < 3600]),
            "timestamp": now,
        }

        if status["errors_last_hour"] > 10:
            status["healthy"] = False
            logger.critical("Health check FAILED: too many errors")

        self.last_check = now
        return status

    def log_error(self, error: str):
        self.errors.append(time.time())
        logger.error(f"Health check error: {error}")


def get_uptime() -> float:
    return time.time()