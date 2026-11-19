"""Request guard, input sniffing and error hygiene.

Threat model: Unmix is a single-user app on 127.0.0.1. What it must withstand
is not a remote attacker on the network, but *web pages open in the same
browser* and *untrusted audio files*:

* CSRF: any page can make the browser POST a multipart form to 127.0.0.1
  (a "simple request", no CORS preflight). -> Origin check on unsafe methods.
* DNS rebinding: a page on attacker.example re-points its domain to 127.0.0.1
  and then reads the API as same-origin. -> Host header allowlist.
* Disk exhaustion: the multipart parser spools the whole body to a temp file
  before the endpoint can check its size. -> size cap enforced while streaming.
* Malicious media: ffmpeg probes input formats; playlist/concat style demuxers
  can be steered into reading other local files. -> we identify the container
  from its magic bytes and force that one demuxer (audio_io.decode_to_wav).
* Information leaks: exception text can contain local paths. -> clients get a
  short message and an error id; details go to the server log only.
"""

from __future__ import annotations

import hmac
import logging
import re
import uuid
from pathlib import Path
from typing import Iterable

log = logging.getLogger("unmix.security")

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
LOCAL_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")

SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-frame-options", b"DENY"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"permissions-policy", b"camera=(), geolocation=(), microphone=()"),
    # API responses are data, never documents: forbid everything if one is ever opened directly.
    (b"content-security-policy", b"default-src 'none'; frame-ancestors 'none'"),
]
DOCS_CSP = b"default-src 'self'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'none'"


def _host_only(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("["):  # [::1]:8000
        return host[: host.find("]") + 1]
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


ACCESS_KEY_HEADER = "x-unmix-key"
MIN_ACCESS_KEY_LENGTH = 24


class GuardMiddleware:
    """Pure ASGI middleware (works with streaming responses and SSE)."""

    def __init__(self, app, *, allowed_hosts: Iterable[str], allowed_origins: Iterable[str], max_body_bytes: int, access_key: str = ""):
        self.app = app
        self.allowed_hosts = {h.lower() for h in allowed_hosts}
        self.allowed_origins = {o.rstrip("/").lower() for o in allowed_origins}
        self.max_body = max_body_bytes
        # Remote mode: the server is reachable through a public tunnel URL, so
        # every request (and WebSocket) must prove it comes from the owner's proxy.
        if access_key and len(access_key) < MIN_ACCESS_KEY_LENGTH:
            raise ValueError(f"UNMIX_REMOTE_KEY must be at least {MIN_ACCESS_KEY_LENGTH} characters.")
        self.access_key = access_key.encode() if access_key else b""

    def _key_ok(self, headers: dict[str, str]) -> bool:
        if not self.access_key:
            return True
        given = headers.get(ACCESS_KEY_HEADER, "").encode("latin-1", "replace")
        return hmac.compare_digest(given, self.access_key)

    def _origin_ok(self, origin: str) -> bool:
        o = origin.rstrip("/").lower()
        return o in self.allowed_origins or bool(LOCAL_ORIGIN_RE.match(o))

    async def _reject(self, send, status: int, detail: str) -> None:
        body = ('{"detail":' + _json_str(detail) + "}").encode()
        await send({"type": "http.response.start", "status": status, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *SECURITY_HEADERS]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            return await self._websocket(scope, receive, send)
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}

        host = _host_only(headers.get("host", ""))
        if host not in self.allowed_hosts:
            log.warning("rejected request with Host %r", headers.get("host"))
            return await self._reject(send, 421, "Unknown host. Unmix only answers on localhost.")
        if not self._key_ok(headers):
            log.warning("rejected %s %s without a valid access key", scope.get("method"), scope.get("path"))
            return await self._reject(send, 401, "Access key missing or wrong.")

        method = scope["method"].upper()
        origin = headers.get("origin")
        if method not in SAFE_METHODS:
            # Browsers always send Origin on cross-site unsafe requests. No Origin = curl/scripts.
            if origin is not None and not self._origin_ok(origin):
                log.warning("rejected %s %s from origin %r", method, scope.get("path"), origin)
                return await self._reject(send, 403, "Cross-site request refused.")
            if headers.get("sec-fetch-site") == "cross-site":
                return await self._reject(send, 403, "Cross-site request refused.")

        # Body cap: refuse early from Content-Length, and count while streaming
        # (chunked uploads have no length).
        if method not in SAFE_METHODS:
            length = headers.get("content-length")
            if length is not None:
                try:
                    if int(length) > self.max_body:
                        return await self._reject(send, 413, f"Upload is too large (limit {self.max_body // (1024 * 1024)} MB).")
                except ValueError:
                    return await self._reject(send, 400, "Invalid Content-Length.")
            received = 0
            too_big = False

            async def limited_receive():
                nonlocal received, too_big
                message = await receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > self.max_body:
                        too_big = True
                        raise _BodyTooLarge()
                return message
        else:
            limited_receive = receive  # type: ignore[assignment]
            too_big = False

        def too_big_flag() -> bool:
            return method not in SAFE_METHODS and too_big

        path = scope.get("path", "")
        is_docs = path in ("/docs", "/redoc", "/openapi.json") or path.startswith("/docs/")
        response_started = False

        async def send_with_headers(message):
            nonlocal response_started
            if too_big_flag():
                # The cap tripped mid-body. Frameworks may turn the aborted read
                # into a generic 400; answer with an honest 413 instead.
                if message["type"] == "http.response.start" and not response_started:
                    response_started = True
                    await self._reject(send, 413, f"Upload is too large (limit {self.max_body // (1024 * 1024)} MB).")
                return
            if message["type"] == "http.response.start":
                response_started = True
                existing = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in SECURITY_HEADERS if k not in existing]
                if is_docs:
                    extra = [(k, DOCS_CSP if k == b"content-security-policy" else v) for k, v in extra]
                message = {**message, "headers": [*message.get("headers", []), *extra]}
            await send(message)

        try:
            await self.app(scope, limited_receive, send_with_headers)
        except _BodyTooLarge:
            if not response_started:
                await self._reject(send, 413, f"Upload is too large (limit {self.max_body // (1024 * 1024)} MB).")


    async def _websocket(self, scope, receive, send):
        """Browsers do not apply CORS to WebSockets: any page can open one to
        localhost (cross-site WebSocket hijacking). Same Host and Origin rules
        as for unsafe HTTP methods, but an Origin is *required*: a browser
        always sends one, and only a browser has a microphone to stream."""
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        host_ok = _host_only(headers.get("host", "")) in self.allowed_hosts
        origin = headers.get("origin")
        origin_ok = origin is not None and self._origin_ok(origin)
        if not (host_ok and origin_ok and self._key_ok(headers)):
            log.warning("rejected websocket %s (host %r, origin %r)", scope.get("path"), headers.get("host"), origin)
            message = await receive()  # websocket.connect
            if message["type"] == "websocket.connect":
                await send({"type": "websocket.close", "code": 1008})
            return
        return await self.app(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


def _json_str(s: str) -> str:
    import json

    return json.dumps(s)


# --------------------------------------------------------------------------- #
# Input sniffing
# --------------------------------------------------------------------------- #

# container -> ffmpeg demuxer name. Anything else is refused before ffmpeg sees it.
def sniff_audio_format(path: Path) -> str | None:
    with open(path, "rb") as f:
        head = f.read(64)
    if len(head) < 12:
        return None
    if head[:4] in (b"RIFF", b"RF64", b"BW64") and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"FORM" and head[8:12] in (b"AIFF", b"AIFC"):
        return "aiff"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:4] == b"\x1a\x45\xdf\xa3":  # EBML: WebM / Matroska (browser recordings)
        return "matroska"
    if head[4:8] == b"ftyp":  # ISO BMFF: mp4 / m4a / mov (Safari recordings)
        return "mov"
    if head[:3] == b"ID3":
        return "mp3"  # ID3v2 tag; an ADTS-AAC file with ID3 would fail cleanly, not unsafely
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        layer = (head[1] >> 1) & 0x3
        return "aac" if layer == 0 else "mp3"  # ADTS has layer bits 00
    return None


# --------------------------------------------------------------------------- #
# Error hygiene
# --------------------------------------------------------------------------- #

_PATH_RE = re.compile(r"([A-Za-z]:[\\/]|\\\\|/(?:home|Users|tmp|var|mnt|opt|root)/)[^\s'\"|:,;]*")


def scrub(text: str) -> str:
    """Remove filesystem paths from a message that will leave the server."""
    return _PATH_RE.sub("<path>", text)


def public_error(prefix: str, exc: BaseException) -> tuple[str, str]:
    """(message for the client, error id). Logs the full exception under the id."""
    error_id = uuid.uuid4().hex[:8]
    log.error("%s [error %s]: %s: %s", prefix, error_id, type(exc).__name__, exc, exc_info=exc)
    return f"{prefix}. Details are in the server log (error {error_id}).", error_id
