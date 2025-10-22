from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class HttpConfig:
    timeout: float = 20.0
    user_agent: str = "trading-engine/1.0 (+https://example.com)"
    # Basic per-host rate limit. Example: 4 rps -> min_interval = 0.25s
    requests_per_second: Optional[float] = None
    # Retry policy
    total_retries: int = 3
    backoff_factor: float = 0.5
    status_forcelist: tuple[int, ...] = (429, 500, 502, 503, 504)
    allowed_methods: frozenset[str] = frozenset(
        {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"}
    )


class HttpClient:
    """Thin wrapper around requests.Session with sane defaults.

    - Global default timeout
    - Retries with backoff on transient errors (429, 5xx)
    - Optional per-host rate limiting (token-bucket-ish via min interval)
    - Minimal surface: get, post, get_json
    """

    def __init__(self, config: Optional[HttpConfig] = None) -> None:
        self.config = config or HttpConfig()
        self._session = requests.Session()
        retry = Retry(
            total=self.config.total_retries,
            read=self.config.total_retries,
            connect=self.config.total_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=self.config.status_forcelist,
            allowed_methods=self.config.allowed_methods,
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update({"User-Agent": self.config.user_agent})
        # Rate limiting state
        self._lock = threading.Lock()
        self._host_next_allowed: Dict[str, float] = {}

    def close(self) -> None:
        self._session.close()

    @contextmanager
    def session(self):
        try:
            yield self
        finally:
            self.close()

    def _rate_limit_sleep(self, host: str) -> None:
        rps = self.config.requests_per_second
        if not rps or rps <= 0:
            return
        min_interval = 1.0 / rps
        with self._lock:
            now = time.monotonic()
            next_allowed = self._host_next_allowed.get(host, 0.0)
            sleep_for = next_allowed - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            # After sleeping, update next window
            self._host_next_allowed[host] = max(now, next_allowed) + min_interval

    def _prepare_kwargs(self, kwargs: dict) -> dict:
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = self.config.timeout
        return kwargs

    def get(self, url: str, **kwargs) -> requests.Response:
        host = _host_from_url(url)
        self._rate_limit_sleep(host)
        kwargs = self._prepare_kwargs(kwargs)
        # Inject Authorization header for Polygon if API key is present
        try:
            if host.endswith("api.polygon.io"):
                import os as _os

                key = _os.environ.get("POLYGON_API_KEY", "")
                if key:
                    hdrs = dict(kwargs.get("headers", {}))
                    if "Authorization" not in {k.title(): v for k, v in hdrs.items()}:
                        hdrs["Authorization"] = f"Bearer {key}"
                    kwargs["headers"] = hdrs
        except Exception:
            pass
        return self._session.get(url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        host = _host_from_url(url)
        self._rate_limit_sleep(host)
        kwargs = self._prepare_kwargs(kwargs)
        try:
            if host.endswith("api.polygon.io"):
                import os as _os

                key = _os.environ.get("POLYGON_API_KEY", "")
                if key:
                    hdrs = dict(kwargs.get("headers", {}))
                    if "Authorization" not in {k.title(): v for k, v in hdrs.items()}:
                        hdrs["Authorization"] = f"Bearer {key}"
                    kwargs["headers"] = hdrs
        except Exception:
            pass
        return self._session.post(url, **kwargs)

    def get_json(self, url: str, **kwargs) -> Any:
        resp = self.get(url, **kwargs)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return None


def _host_from_url(url: str) -> str:
    try:
        # Avoid importing urlparse globally for tiny helper
        from urllib.parse import urlsplit

        return urlsplit(url).netloc
    except Exception:
        return ""
