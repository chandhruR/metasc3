from __future__ import annotations

import os

from server.main import app

__all__ = ["app", "main"]


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", os.environ.get("OPENENV_PORT", "7860")))
    uvicorn.run("server.main:app", host=host, port=port, factory=False)


if __name__ == "__main__":
    main()
