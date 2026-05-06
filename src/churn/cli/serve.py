"""Console entry point: launch the FastAPI inference server."""

from __future__ import annotations

import os

import uvicorn

from churn.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    configure_logging()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "false").lower() == "true"
    logger.info("Starting churn API", extra={"host": host, "port": port, "reload": reload})
    uvicorn.run("churn.api.app:app", host=host, port=port, reload=reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
