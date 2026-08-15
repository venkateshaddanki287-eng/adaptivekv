"""Lightweight HTTP server for serving AdaptiveKV research dashboard and benchmark API."""

from __future__ import annotations

import http.server
import json
import socketserver
from pathlib import Path

PORT = 8501
DASHBOARD_DIR = Path(__file__).parent
RESULTS_DIR = Path(__file__).parent.parent / "research" / "results"


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            records = []
            if RESULTS_DIR.exists():
                for json_file in sorted(RESULTS_DIR.glob("*.json")):
                    try:
                        with open(json_file, encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                records.extend(data)
                            elif isinstance(data, dict):
                                records.append(data)
                    except Exception:
                        pass

            self.wfile.write(json.dumps({"results": records}).encode("utf-8"))
            return

        super().do_GET()


class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def main() -> None:
    port = PORT
    httpd = None
    for try_port in (8501, 8502, 8503, 8504, 8505):
        try:
            httpd = ReusableTCPServer(("", try_port), DashboardHandler)
            port = try_port
            break
        except OSError:
            continue

    if httpd is None:
        print("[Error] Could not bind to any port in range 8501-8505.")
        return

    with httpd:
        print(f"==================================================")
        print(f" AdaptiveKV Dashboard running at: http://localhost:{port}")
        print(f" Reading benchmark data from: {RESULTS_DIR}")
        print(f" Press Ctrl+C to stop.")
        print(f"==================================================")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard server stopped.")


if __name__ == "__main__":
    main()

