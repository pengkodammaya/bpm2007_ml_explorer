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
    sleep_s: float = 0.35,
    allow_404: bool = False,
    max_retries: int = 8,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if allow_404 and resp.status_code == 404:
                return {}
            # Retry on 429 (rate limit) and 5xx (server errors)
            if resp.status_code == 429 or resp.status_code >= 500:
                wait = min(2 ** attempt * 3 + 5, 120)  # 8, 11, 17, 29, 53, 101, 120, 120s
                if attempt < max_retries - 1:
                    print(f"[RETRY] {resp.status_code} on attempt {attempt+1}/{max_retries}, waiting {wait}s...",
                          flush=True)
                    time.sleep(wait)
                    continue
            resp.raise_for_status()
            time.sleep(sleep_s)
            return resp.json()
        except requests.RequestException as e:
            last_exc = e
            if attempt < max_retries - 1:
                wait = min(2 ** attempt * 3 + 5, 120)
                time.sleep(wait)
            else:
                raise HTTPFetchError(f"Failed GET {url} params={params}: {e}") from e
    raise HTTPFetchError(f"Failed GET {url} params={params} after {max_retries} retries: {last_exc}") from last_exc


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default
