"""Run the workbench: ``python -m scanlate``.

``SCANLATE_DB=scanlate.db`` persists to disk, ``SCANLATE_BACKEND=opus`` swaps
the deterministic phrase table for the neural model.
"""
from __future__ import annotations

import os

import uvicorn

from .api.app import create_app


def main() -> None:
    uvicorn.run(create_app(), host=os.environ.get("SCANLATE_HOST", "127.0.0.1"),
                port=int(os.environ.get("SCANLATE_PORT", "8000")))


if __name__ == "__main__":
    main()
