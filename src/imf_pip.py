from __future__ import annotations

import json
import requests
import pandas as pd
from typing import Any
from config import USER_AGENT

HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}


def fetch_url_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    r = requests.get(url, params=params, headers=HEADERS, timeout=90)
    r.raise_for_status()
    return r.json()


def save_raw_imf_response(url: str, output_path: str, params: dict[str, Any] | None = None) -> None:
    payload = fetch_url_json(url, params=params)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def generic_observation_extract(payload: dict[str, Any]) -> pd.DataFrame:
    datasets = payload.get("data", {}).get("dataSets", [])
    if datasets:
        observations = datasets[0].get("observations", {})
        rows = []
        for obs_key, obs_val in observations.items():
            row = {"obs_key": obs_key}
            if isinstance(obs_val, list) and obs_val:
                row["value"] = obs_val[0]
            else:
                row["value"] = None
            rows.append(row)
        return pd.DataFrame(rows)

    if isinstance(payload, dict):
        flat = flatten_dict(payload)
        return pd.DataFrame([flat])

    return pd.DataFrame()
