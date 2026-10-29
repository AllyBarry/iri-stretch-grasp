'''
Web viewer for the simulation, using only the standard library.

    GET  /                  the viewer page (static/)
    GET  /api/world         the static world: map, locations, robot geometry
    GET  /api/events?since  logged events after sequence number `since`
    GET  /api/events.jsonl  the whole in-memory log, as a download
    GET  /api/stream?since  server-sent events: {"state": ..., "events": [...]} ~20 times a second
    POST /api/command       {"cmd": "goto", "location": "Fridge"}, ... (see SimController)
'''
import json
import logging
import mimetypes
import threading
import time
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from stretch_sim.controller import CommandError, SimController

log = logging.getLogger("stretch_sim.web")


def static_dir() -> Path:
    source = Path(__file__).resolve().parent / "static"
    if source.is_dir():
        return source
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("stretch_floor_perception")) / "sim_web"


class _Handler(BaseHTTPRequestHandler):
    controller: SimController = None
    stream_hz: float = 20.0
    static_root: Path = None

    def log_message(self, fmt, *args):
        log.debug("%s - %s", self.address_string(), fmt % args)

    # Responses
    def _send(self, status: int, body: bytes, content_type: str, extra_headers: Optional[dict] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(data).encode("utf-8"), "application/json")

    def _since(self, query) -> int:
        try:
            return int(query.get("since", ["0"])[0])
        except ValueError:
            return 0

    # Routes
    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/api/world":
            return self._json(self.controller.world())
        if url.path == "/api/state":
            return self._json(self.controller.state())
        if url.path == "/api/events":
            return self._json([asdict(e) for e in self.controller.sim.log.since(self._since(query))])
        if url.path == "/api/events.jsonl":
            body = "".join(json.dumps(asdict(e)) + "\n" for e in self.controller.sim.log.events).encode("utf-8")
            return self._send(HTTPStatus.OK, body, "application/x-ndjson",
                              {"Content-Disposition": 'attachment; filename="events.jsonl"'})
        if url.path == "/api/stream":
            return self._stream(self._since(query))
        return self._static(url.path)

    def do_POST(self):
        if urlparse(self.path).path != "/api/command":
            return self._json({"ok": False, "message": "not found"}, HTTPStatus.NOT_FOUND)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            cmd = json.loads(self.rfile.read(length) or b"{}")
            message = self.controller.execute(cmd)
            return self._json({"ok": True, "message": message})
        except CommandError as err:
            return self._json({"ok": False, "message": str(err)}, HTTPStatus.BAD_REQUEST)
        except (ValueError, TypeError) as err:
            return self._json({"ok": False, "message": f"bad request: {err}"}, HTTPStatus.BAD_REQUEST)

    def _static(self, path: str):
        root = self.static_root.resolve()
        target = (root / ("index.html" if path in ("", "/") else path.lstrip("/"))).resolve()
        if root not in target.parents and target != root or not target.is_file():
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type.endswith("javascript"):
            content_type += "; charset=utf-8"
        return self._send(HTTPStatus.OK, target.read_bytes(), content_type)

    def _stream(self, since: int):
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        period = 1.0 / self.stream_hz
        try:
            while not self.server.stopping.is_set():
                events = self.controller.sim.log.since(since)
                if events:
                    since = events[-1].seq
                payload = {"state": self.controller.state(), "events": [asdict(e) for e in events]}
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(period)
        except (BrokenPipeError, ConnectionResetError):
            pass


class WebViewer:
    def __init__(self, controller: SimController, host: str = "127.0.0.1", port: int = 8765):
        handler = type("Handler", (_Handler,), {"controller": controller, "static_root": static_dir()})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.httpd.stopping = threading.Event()
        self.controller = controller

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"

    def serve_forever(self) -> None:
        self.httpd.serve_forever(poll_interval=0.2)

    def start_background(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="web-viewer", daemon=True)
        thread.start()
        return thread

    def shutdown(self) -> None:
        self.httpd.stopping.set()
        self.httpd.shutdown()
        self.httpd.server_close()
