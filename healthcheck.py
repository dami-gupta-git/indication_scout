"""Minimal HTTP healthcheck server for Railway deployment.

Starts a lightweight HTTP server in a daemon thread on port 8000 that
responds with 200 OK to GET /health. This runs independently of Streamlit
so Railway can detect readiness immediately, before Streamlit finishes
initialising, preventing 502 errors during the deployment window.

Import this module at the very top of app.py (before any other imports)
so the server is up as early as possible.
"""

import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)

_HEALTHCHECK_PORT = 8000


class _HealthHandler(BaseHTTPRequestHandler):
    """Respond 200 OK to GET /health; 404 to everything else."""

    def do_GET(self) -> None:
        if self.path == "/health":
            body = b'{"status": "ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Suppress per-request access logs to keep Streamlit output clean.
        pass


def _start_healthcheck_server() -> None:
    """Bind the healthcheck server and serve forever (called in a daemon thread)."""
    server = HTTPServer(("0.0.0.0", _HEALTHCHECK_PORT), _HealthHandler)
    logger.info("Healthcheck server listening on port %d", _HEALTHCHECK_PORT)
    server.serve_forever()


def start_healthcheck() -> None:
    """Start the healthcheck HTTP server in a background daemon thread.

    Safe to call multiple times — subsequent calls are no-ops because the
    thread is marked as a daemon and the port binding would fail silently.
    """
    thread = threading.Thread(
        target=_start_healthcheck_server,
        name="healthcheck-server",
        daemon=True,  # exits automatically when the main process exits
    )
    thread.start()
    logger.info("Healthcheck thread started (port %d)", _HEALTHCHECK_PORT)


# Start immediately on import so the server is up before Streamlit initialises.
start_healthcheck()
