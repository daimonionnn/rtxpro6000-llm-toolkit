#!/usr/bin/env python3
"""Serve the chat UI on http://127.0.0.1:5173.

    python3 ui/serve.py [port]

The page talks to the SGLang server at http://127.0.0.1:8090 directly; SGLang
reflects the Origin header, so no proxy is needed. Point it elsewhere with
?api=http://host:port. Serving over http rather than opening the file with
file:// matters: a file:// page sends Origin: null and the request is refused.
"""
import functools, http.server, socketserver, sys, webbrowser
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5173
ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


socketserver.TCPServer.allow_request_reuse = True
with socketserver.TCPServer(("127.0.0.1", PORT),
                            functools.partial(Handler, directory=str(ROOT))) as httpd:
    url = f"http://127.0.0.1:{PORT}/"
    print(f"chat UI on {url}   (Ctrl+C to stop)")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    httpd.serve_forever()
