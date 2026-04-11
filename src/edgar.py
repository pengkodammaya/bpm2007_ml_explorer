from __future__ import annotations

import re
import requests
import pandas as pd
from typing import Any
from config import USER_AGENT

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}


def clean_company_name(name: str | None) -> str:
    if not name:
        return ""
    name = name.upper().strip()
    name = re.sub(r"[^A-Z0-9 ]+", " ", name)
    name = re.sub(r"\s+", " ", name)
    return name


def fetch_company_tickers() -> pd.DataFrame:
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    payload = r.json()

    rows: list[dict[str, Any]] = []
    for _, item in payload.items():
        rows.append({
            "cik": str(item.get("cik_str")).zfill(10),
            "ticker": item.get("ticker"),
            "title": item.get("title"),
            "title_clean": clean_company_name(item.get("title")),
        })

    return pd.DataFrame(rows)


def match_entities_to_edgar(entities: pd.DataFrame, tickers: pd.DataFrame) -> pd.DataFrame:
    df = entities.copy()
    if "legal_name" not in df.columns:
        df["legal_name"] = None
    df["legal_name_clean"] = df["legal_name"].map(clean_company_name)

    tickers = tickers.reindex(columns=["cik", "ticker", "title", "title_clean"])
    if tickers.empty:
        df["appears_in_edgar"] = 0
        df["cik"] = None
        df["ticker"] = None
        df["title"] = None
        return df

    merged = df.merge(
        tickers[["cik", "ticker", "title", "title_clean"]],
        left_on="legal_name_clean",
        right_on="title_clean",
        how="left",
    )
    merged["appears_in_edgar"] = merged["cik"].notna().astype(int)
    return merged


def fetch_company_submissions(cik: str) -> dict[str, Any]:
    cik = str(cik).zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()
