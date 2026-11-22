"""Local development entry point.

Run with:  python run.py
"""
from __future__ import annotations

import os

from app import create_app

app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5001"))
    debug = os.getenv("SPAMGUARD_DEBUG", "false").lower() in {"1", "true", "yes"}
    # use_reloader=False avoids loading the model twice during a demo.
    app.run(host="127.0.0.1", port=port, debug=debug, use_reloader=False)
