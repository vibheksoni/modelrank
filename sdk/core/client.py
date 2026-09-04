"""Base HTTP client for the external data sources.

Built on curl_cffi (requests-compatible) with Chrome TLS impersonation:

- DNS is pre-resolved through Python's own resolver (`socket.getaddrinfo`)
  and pinned into libcurl via `CurlOpt.RESOLVE` for every host the tool
  talks to (openrouter.ai, artificialanalysis.ai, models.dev). This keeps
  the Host header and SNI intact while bypassing libcurl's resolver
  entirely - if libcurl's DNS layer breaks (a known curl_cffi pain point),
  the resolved IPs are still used directly.
- `interface` optionally binds outgoing connections to a specific local
  NIC / source IP (`CurlOpt.INTERFACE`).
- No custom User-Agent is set: impersonation injects a real browser UA.
"""

from __future__ import annotations

import socket
from typing import Iterable, Optional

from curl_cffi import requests as cr
from curl_cffi.const import CurlOpt
from curl_cffi.requests.exceptions import RequestException  # noqa: F401 (re-export)

from sdk.core.constants import BASE_HEADERS, TIMEOUT

# Hosts the client resolves + pins by default.
DEFAULT_HOSTS = ("openrouter.ai", "artificialanalysis.ai", "models.dev")

IMPersonate = "chrome"


class OpenRouterClient:
    """Session-based client for OpenRouter with DNS pinning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = TIMEOUT,
        impersonate: str = "chrome",
        interface: Optional[str] = None,
        hosts: Iterable[str] = DEFAULT_HOSTS,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.impersonate = impersonate
        self.interface = interface
        self.resolved: dict[str, list[str]] = {}
        for host in hosts:
            ips = self._resolve(host)
            if ips:
                self.resolved[host] = ips
        self.session = self._build_session()

    # ── construction ────────────────────────────────────────────────────

    @staticmethod
    def _resolve(host: str) -> list[str]:
        """Resolve via Python's resolver; fall back to an empty list."""
        try:
            return sorted({i[4][0] for i in socket.getaddrinfo(host, 443, socket.AF_INET)})
        except OSError:
            return []

    def _build_session(self) -> cr.Session:
        session = cr.Session(impersonate=self.impersonate)
        session.headers.update(BASE_HEADERS)
        if self.api_key:
            session.headers["Authorization"] = f"Bearer {self.api_key}"
        pins = []
        for host, ips in self.resolved.items():
            for ip in ips:
                pins.append(f"{host}:443:{ip}")
        if pins:
            session.curl.setopt(CurlOpt.RESOLVE, pins)
        return session

    # ── requests ────────────────────────────────────────────────────────

    def get(self, url: str, params: Optional[dict] = None, timeout: Optional[float] = None) -> cr.Response:
        kwargs = {}
        if self.interface:
            kwargs["interface"] = self.interface
        return self.session.get(url, params=params, timeout=timeout or self.timeout, **kwargs)

    def get_json(self, url: str, params: Optional[dict] = None, timeout: Optional[float] = None):
        resp = self.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "OpenRouterClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()