from __future__ import annotations

import uvicorn

from starling.config import settings


def main() -> None:
    uvicorn.run(
        "starling.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
