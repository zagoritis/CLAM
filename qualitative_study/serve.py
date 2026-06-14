#!/usr/bin/env python3
"""
Tiny static file server that supports HTTP Range requests.

Python's stock `python -m http.server` ignores the `Range:` header and always returns
the whole file with `200 OK`, which means the browser cannot seek/scrub an HTML5 <video>
to an arbitrary timestamp. This server answers Range requests with `206 Partial Content`
so video seeking works.

Usage (from the qualitative_study/ folder, or anywhere):
    python serve.py            # serves this script's folder on http://localhost:8000
    python serve.py 8080       # custom port

Then open  http://localhost:8000/annotate.html
"""
import functools
import os
import re
import socket
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))


def port_in_use(port):
    """True if something already accepts connections on this port.

    An active connect check, not a bind attempt: on Windows, SO_REUSEADDR lets a new
    server silently co-bind a port an old server (e.g. `python -m http.server`) is
    already holding, so a failed/succeeded bind can't tell us the port is really free.
    Connecting to it can.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


class RangeRequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # enables keep-alive; we always send Content-Length

    def end_headers(self):
        # Advertise range support on every response so browsers know seeking is allowed.
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        range_header = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not range_header or os.path.isdir(path):
            return super().send_head()  # normal full response / directory listing

        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None

        size = os.fstat(f.fileno()).st_size
        m = re.match(r"bytes=(\d*)-(\d*)\s*$", range_header)
        if not m or (m.group(1) == "" and m.group(2) == ""):
            f.close()
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Range header")
            return None

        start_s, end_s = m.group(1), m.group(2)
        if start_s == "":                       # suffix range: last N bytes
            length = min(int(end_s), size)
            start, end = size - length, size - 1
        else:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        end = min(end, size - 1)

        if start > end or start >= size:
            f.close()
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(length))
        self.end_headers()
        f.seek(start)
        self._range_remaining = length
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_range_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        self._range_remaining = None
        bufsize = 64 * 1024
        while remaining > 0:
            chunk = source.read(min(bufsize, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)


def main():
    requested = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = functools.partial(RangeRequestHandler, directory=HERE)

    # Skip ports that already have a server answering on them (see port_in_use).
    port = requested
    while port < requested + 20 and port_in_use(port):
        port += 1

    if port != requested:
        print("=" * 70)
        print(f"!! Port {requested} is already serving -- most likely an old")
        print(f"!! `python -m http.server` is still running there. That server does")
        print(f"!! NOT support video seeking. Best: stop it (Ctrl+C in its window) and")
        print(f"!! re-run this script so the URL stays {requested}.")
        print(f"!! For now, serving on {port} instead -- open the URL below, not {requested}.")
        print("=" * 70)

    try:
        httpd = ThreadingHTTPServer(("", port), handler)
    except OSError as e:
        print(f"ERROR: could not bind port {port}: {e}")
        sys.exit(1)

    with httpd:
        print(f"Serving {HERE}")
        print(f"Open  http://localhost:{port}/annotate.html   (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
