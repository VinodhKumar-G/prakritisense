#!/usr/bin/env python3
"""
PrakritiSense local server
───────────────────────────
Serves the app (same as `python3 -m http.server`) AND accepts POST
requests from the app's export buttons, writing CSV/JSON exports
directly into ./data/ inside this Codespace — no manual download
and re-upload needed.

Run:
    python3 server.py
    python3 server.py 8000   (optional explicit port, default 8000)
"""

import http.server
import os
import sys
import json

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path not in ("/save-csv", "/save-json"):
            self.send_response(404)
            self._cors()
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")

        try:
            payload = json.loads(body)
            filename = os.path.basename(payload.get("filename", "session.dat"))
            content = payload.get("content", "")
        except Exception as e:
            self.send_response(400)
            self._cors()
            self.end_headers()
            self.wfile.write(str(e).encode())
            return

        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"[server] saved {filename}  ({len(content)} bytes)  ->  data/{filename}")

        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok", "path": f"data/{filename}"}).encode())

    def log_message(self, fmt, *args):
        # Keep terminal output minimal — only print save confirmations above
        pass


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"PrakritiSense running at http://localhost:{PORT}")
    print(f"Exports will be saved directly into: {DATA_DIR}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")