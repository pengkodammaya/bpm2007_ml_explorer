from __future__ import annotations

import time
import requests
from typing import Any


class HTTPFetchError(RuntimeError):
    pass


def get_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    sleep_s: float = 0.2,
    allow_404: bool = False,
) -> dict[str, Any]:
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        if allow_404 and resp.status_code == 404:
            return {}
        resp.raise_for_status()
        time.sleep(sleep_s)
        return resp.json()
    except requests.RequestException as e:
        raise HTTPFetchError(f"Failed GET {url} params={params}: {e}") from e


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default
