# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal completion server for the startup benchmark's subprocess tests."""

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def serve(port: int, request_log: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            if self.path == "/v1/models":
                self.wfile.write(b'{"data":[{"id":"test"}]}')

        def do_POST(self):
            with request_log.open("a") as log:
                print(os.environ["VLLM_CACHE_ROOT"], file=log)
            self.send_response(200)
            self.end_headers()
            # Each restart generates different text. Only its arrival is timed.
            chunk = {"choices": [{"index": 0, "text": str(os.getpid())}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--request-log", type=Path, required=True)
    args = parser.parse_args()
    serve(args.port, args.request_log)
