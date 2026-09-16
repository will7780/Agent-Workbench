"""Two loopback ThreadingHTTPServers sharing a WebAppService and runtime.

POST /api/runs {user_request, thread_id?} -> 202 operation
POST /api/runs/:id/resume {interaction_id, response} -> 202 operation
POST /api/runs/:id/cancel {} -> 202 operation
GET /api/operations/:id -> transport status and projected result
GET /api/runs -> bounded, redacted navigation summaries
GET /api/runs/:id -> bounded, redacted runtime evidence projection

Mutation requests require application/json and the exact page Origin (including
port). The two pages navigate between ports, never make cross-origin API calls.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .service import WebAppService, WebError, public_data

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_BODY_BYTES = 65536
_RUN = re.compile(r"/api/runs/([A-Za-z0-9][A-Za-z0-9_.:-]{0,127})(?:/(resume|cancel))?\Z")
_OPERATION = re.compile(r"/api/operations/([A-Za-z0-9_]{1,128})\Z")
_ASSETS = {"/styles.css": ("styles.css", "text/css"), "/app.js": ("app.js", "text/javascript"),
           "/inspection.js": ("inspection.js", "text/javascript")}


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


class WorkbenchHandler(BaseHTTPRequestHandler):
    service: WebAppService
    page: str
    peer_port: int
    server_version = "AgentWorkbench"
    sys_version = ""

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(5)

    def log_message(self, *args: Any) -> None:
        pass  # Request paths and raw parser errors may contain sensitive text.

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:
        self._json(code, {"error_type": "http_request_rejected"})

    def _reply(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; "
                         "connect-src 'self'; img-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def _json(self, status: int, value: Any) -> None:
        body = json.dumps(public_data(value), ensure_ascii=True, allow_nan=False).encode("utf-8")
        self._reply(status, body, "application/json")

    def _check_origin(self, *, mutation: bool = False) -> None:
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host_values = self.headers.get_all("Host", [])
        if len(host_values) != 1 or host_values[0] not in hosts:
            raise WebError("invalid_host", 403)
        origins = self.headers.get_all("Origin", [])
        expected = "http://" + host_values[0]
        if (mutation and not origins) or (origins and origins != [expected]):
            raise WebError("same_origin_required", 403)
        fetch_site = self.headers.get("Sec-Fetch-Site")
        if mutation and fetch_site not in {None, "same-origin", "none"}:
            raise WebError("same_origin_required", 403)
        if self.path.startswith("/api/") and fetch_site in {"cross-site", "same-site"}:
            raise WebError("same_origin_required", 403)

    def _body(self) -> dict:
        if self.headers.get_all("Transfer-Encoding"):
            raise WebError("transfer_encoding_forbidden")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,9}", lengths[0]):
            raise WebError("invalid_content_length", 411)
        length = int(lengths[0])
        if length > MAX_BODY_BYTES:
            raise WebError("request_too_large", 413)
        if length == 0:
            raise WebError("json_object_required")
        types = self.headers.get_all("Content-Type", [])
        if len(types) != 1 or types[0].split(";", 1)[0].strip().lower() != "application/json":
            raise WebError("json_required", 415)
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete_body")
            payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                                 parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid_number")))
        except (UnicodeError, ValueError, RecursionError, TimeoutError):
            raise WebError("invalid_json") from None
        if not isinstance(payload, dict):
            raise WebError("json_object_required")
        return payload

    def do_GET(self) -> None:
        try:
            self._check_origin()
            parsed = urlsplit(self.path)
            if parsed.scheme or parsed.netloc:
                raise WebError("invalid_path")
            path = parsed.path
            if path == "/api/health":
                return self._json(200, {"status": "ok", "page": self.page})
            if path == "/api/meta":
                return self._json(200, {**self.service.metadata(), "page": self.page, "peer_port": self.peer_port})
            if path == "/api/runs":
                return self._json(200, self.service.list_runs())
            if path == "/api/catalogue":
                return self._json(200, self.service.tool_catalogue())
            if path.startswith("/api/catalogue/"):
                return self._json(200, self.service.tool_catalogue(path[len("/api/catalogue/"):]))
            match = _OPERATION.fullmatch(path)
            if match:
                return self._json(200, self.service.get_operation(match[1]))
            match = _RUN.fullmatch(path)
            if match and match[2] is None:
                return self._json(200, self.service.get_run(match[1]))
            if path in {"/", "/index.html"}:
                asset, mime = ("chat.html" if self.page == "chat" else "diagnostics.html"), "text/html"
            elif path in _ASSETS:
                asset, mime = _ASSETS[path]
            else:
                raise WebError("not_found", 404)
            target = (STATIC_DIR / asset).resolve(strict=True)
            if target.parent != STATIC_DIR.resolve():
                raise WebError("not_found", 404)
            return self._reply(200, target.read_bytes(), mime)
        except WebError as exc:
            self._json(exc.status, {"error_type": exc.code})
        except Exception:
            self._json(500, {"error_type": "web_request_failed"})

    def do_POST(self) -> None:
        try:
            self._check_origin(mutation=True)
            body = self._body()
            if self.path == "/api/runs":
                return self._json(202, self.service.start(body))
            match = _RUN.fullmatch(self.path)
            if match and match[2] == "resume":
                return self._json(202, self.service.resume(match[1], body))
            if match and match[2] == "cancel":
                if body:
                    raise WebError("unsupported_request_field")
                return self._json(202, self.service.cancel(match[1]))
            raise WebError("not_found", 404)
        except WebError as exc:
            self._json(exc.status, {"error_type": exc.code})
        except Exception:
            self._json(500, {"error_type": "web_request_failed"})


def make_handler(service: WebAppService, *, page: str = "chat", peer_port: int = 8785) -> type:
    if page not in {"chat", "diagnostics"}:
        raise ValueError("invalid_page")

    class BoundHandler(WorkbenchHandler):
        pass

    BoundHandler.service, BoundHandler.page, BoundHandler.peer_port = service, page, peer_port
    return BoundHandler


def create_servers(runtime: Any, *, host: str = "127.0.0.1", chat_port: int = 8786,
                   diagnostics_port: int = 8785, offline: bool | None = None,
                   model_label: str | None = None) -> tuple[ThreadingHTTPServer, ThreadingHTTPServer]:
    """Bind both listeners, but let the caller choose thread/lifecycle ownership.

    server.RequestHandlerClass.service is the shared adapter. Call its close()
    after shutdown/server_close of both servers; runtime shutdown remains host-owned.
    """
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("loopback_only")
    service = WebAppService(runtime, offline=offline, model_label=model_label)
    chat = None
    try:
        chat = ThreadingHTTPServer(("127.0.0.1", chat_port), make_handler(service, page="chat"))
        diagnostics = ThreadingHTTPServer(("127.0.0.1", diagnostics_port), make_handler(service, page="diagnostics"))
    except Exception:
        if chat:
            chat.server_close()
        service.close()
        raise
    chat.RequestHandlerClass.peer_port = diagnostics.server_address[1]
    diagnostics.RequestHandlerClass.peer_port = chat.server_address[1]
    return chat, diagnostics


def run_servers(runtime: Any, **options: Any) -> None:
    """Blocking host entry point; no default runtime or private imports."""
    servers = create_servers(runtime, **options)
    threads = [threading.Thread(target=server.serve_forever, daemon=True) for server in servers]
    for thread in threads:
        thread.start()
    try:
        for thread in threads:
            while thread.is_alive():
                thread.join(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join()
        servers[0].RequestHandlerClass.service.close()
