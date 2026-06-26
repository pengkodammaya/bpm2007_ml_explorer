from __future__ import annotations

import re
import time
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests
from rapidfuzz import fuzz, process

from config import USER_AGENT
from src.edgar import clean_company_name


CTOS_BASE = "https://businessreport.ctoscredit.com.my/oneoffreport_api"
LISTING_URL = CTOS_BASE + "/malaysia-company-listing/{letter}/{page}"

CTOS_COLUMNS = [
    "ctos_name",
    "ctos_url",
    "ctos_listing_letter",
    "ctos_listing_page",
    "ctos_registration_no",
    "ctos_nature_of_business",
    "ctos_registration_date",
    "ctos_state",
]

CTOS_MATCH_COLUMNS = [
    "lei",
    "legal_name",
    "ctos_name",
    "ctos_url",
    "ctos_registration_no",
    "ctos_match_score",
    "ctos_match_method",
    "ctos_registered_malaysia",
]

CTOS_LOCAL_COLUMNS = [
    "ctos_name",
    "ctos_url",
    "ctos_listing_letter",
    "ctos_listing_page",
    "ctos_registration_no",
    "ctos_nature_of_business",
    "ctos_registration_date",
    "ctos_state",
]

_GENERIC_NAME_TOKENS = {
    "SDN", "BHD", "BERHAD", "SENDIRIAN", "LTD", "LIMITED", "PTE",
    "CO", "COMPANY", "CORPORATION", "CORP", "GROUP", "HOLDING", "HOLDINGS",
    "M", "MALAYSIA", "ASIA", "GLOBAL", "INTERNATIONAL",
}

_WEAK_FUZZY_TOKENS = {
    "AGRIBUSINESS", "ASSET", "BANK", "CAPITAL", "CENTRE", "CONSULTANTS",
    "DEVELOPMENT", "DORMITORY", "ENGINEERING", "ENTERPRISE", "FOOD",
    "FRESH", "GUARD", "HEALTHCARE", "HOLIDAY", "INDUSTRIES", "INDUSTRY",
    "INVESTMENT", "LAND", "MANAGEMENT", "MANUFACTURING", "MEDICAL",
    "MINERALS", "RESOURCES", "SECURITY", "SERVICES", "SOLUTIONS",
    "SYSTEMS", "TECH", "TECHNOLOGIES", "TECHNOLOGY", "TOURS", "TRADING",
    "TRAINING", "TRAVEL", "VENTURES", "WEALTH",
}


class _ListingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[dict[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href and "/single-report/malaysia-company/" in href:
            self._href = href
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._href:
            return
        name = unescape(" ".join(self._text)).strip()
        if name:
            href = self._href
            if href.startswith("/"):
                href = CTOS_BASE + href
            self.links.append({"ctos_name": name, "ctos_url": href})
        self._href = None
        self._text = []


def normalize_malaysia_company_name(name: str | None) -> str:
    """Normalize a company name for Malaysia registry/LEI matching."""
    cleaned = clean_company_name(name)
    replacements = [
        (r"\bSDN\s+BHD\b", "SDN BHD"),
        (r"\bSENDIRIAN\s+BERHAD\b", "SDN BHD"),
        (r"\bBHD\b", "BHD"),
        (r"\bBERHAD\b", "BHD"),
        (r"\bPTE\s+LTD\b", "PTE LTD"),
        (r"\bPRIVATE\s+LIMITED\b", "PTE LTD"),
        (r"\bLIMITED\b", "LTD"),
        (r"\bCOMPANY\b", "CO"),
    ]
    for pattern, repl in replacements:
        cleaned = re.sub(pattern, repl, cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _distinctive_token_list(normalized_name: str) -> list[str]:
    return [
        token for token in normalized_name.split()
        if len(token) > 1 and token not in _GENERIC_NAME_TOKENS
    ]


def _is_safe_fuzzy_ctos_match(entity_norm: str, ctos_norm: str) -> bool:
    entity_tokens = _distinctive_token_list(entity_norm)
    ctos_tokens = _distinctive_token_list(ctos_norm)
    if not entity_tokens or not ctos_tokens:
        return False

    entity_set = set(entity_tokens)
    ctos_set = set(ctos_tokens)
    shared = entity_set & ctos_set
    strong_shared = shared - _WEAK_FUZZY_TOKENS
    if not strong_shared:
        return False

    first_entity = entity_tokens[0]
    first_ctos = ctos_tokens[0]
    first_token_aligned = (
        first_entity == first_ctos
        or fuzz.ratio(first_entity, first_ctos) >= 92
    )
    if first_token_aligned:
        return True

    # If the leading token differs, require enough strong overlap that the
    # match is not being driven by generic business activity words.
    return len(strong_shared) >= 2


def load_ctos_company_snapshot(
    path: str | Path,
    *,
    use_duckdb: bool = True,
) -> pd.DataFrame:
    """Load a local CTOS-derived Malaysia company snapshot.

    The crawler snapshot supplied by the user has columns like ``name``,
    ``_letter`` and ``_page``. This converts it to the same shape as the live
    CTOS crawler output so downstream matching treats both sources identically.
    DuckDB is used when installed because it can project parquet columns cheaply;
    pandas is used as a dependency-free fallback.
    """
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(snapshot_path)

    columns = ["name", "_letter", "_page"]
    if use_duckdb:
        try:
            import duckdb  # type: ignore[import-not-found]

            escaped = str(snapshot_path).replace("'", "''")
            query = (
                "select name, _letter, _page "
                f"from read_parquet('{escaped}') "
                "where name is not null"
            )
            raw = duckdb.sql(query).df()
        except ModuleNotFoundError:
            raw = pd.read_parquet(snapshot_path, columns=columns)
    else:
        raw = pd.read_parquet(snapshot_path, columns=columns)

    companies = pd.DataFrame({
        "ctos_name": raw["name"].astype(str).str.strip(),
        "ctos_url": None,
        "ctos_listing_letter": raw.get("_letter"),
        "ctos_listing_page": raw.get("_page"),
        "ctos_registration_no": None,
        "ctos_nature_of_business": None,
        "ctos_registration_date": None,
        "ctos_state": None,
    })
    companies = companies[companies["ctos_name"].str.len() > 0]
    companies = companies.drop_duplicates(
        subset=["ctos_name", "ctos_listing_letter", "ctos_listing_page"],
        keep="first",
    )
    return companies[CTOS_LOCAL_COLUMNS].reset_index(drop=True)


def parse_ctos_listing(html: str) -> pd.DataFrame:
    parser = _ListingParser()
    parser.feed(html)
    if not parser.links:
        return pd.DataFrame(columns=["ctos_name", "ctos_url"])
    return pd.DataFrame(parser.links).drop_duplicates(subset=["ctos_name", "ctos_url"])


def parse_ctos_company_detail(html: str) -> dict[str, Any]:
    text = re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()

    def pick(pattern: str) -> str | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return _clean_detail_field(match.group(1)) if match else None

    return {
        "ctos_registration_no": pick(r"Company Registration No\.\s*([^ ]+\s*/\s*\d+)"),
        "ctos_nature_of_business": pick(r"Nature of Business\s*(.*?)\s*Date of Registration"),
        "ctos_registration_date": pick(r"Date of Registration\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"),
        "ctos_state": pick(r"State\s*(.*?)\s*COMPANY DESCRIPTION"),
    }


def _clean_detail_field(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"<!--|-->", " ", str(value))
    cleaned = re.sub(r"\s+-\s*$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    if not cleaned or cleaned.lower() == "null":
        return None
    return cleaned


def _get_with_retries(
    url: str,
    *,
    timeout: int = 60,
    max_retries: int = 3,
    retry_sleep_s: float = 2.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> requests.Response:
    last_error: requests.RequestException | None = None
    for attempt in range(1, max(max_retries, 1) + 1):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max(max_retries, 1):
                break
            print(
                f"[WARN] CTOS request failed on attempt {attempt}/{max_retries}: {exc}",
                flush=True,
            )
            sleep_fn(retry_sleep_s * attempt)
    assert last_error is not None
    raise last_error


def fetch_ctos_listing_page(
    letter: str,
    page: int,
    *,
    timeout: int = 60,
    max_retries: int = 3,
    retry_sleep_s: float = 2.0,
) -> pd.DataFrame:
    url = LISTING_URL.format(letter=letter.lower(), page=page)
    response = _get_with_retries(
        url,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep_s=retry_sleep_s,
    )
    result = parse_ctos_listing(response.text)
    if not result.empty:
        result["ctos_listing_letter"] = letter.upper()
        result["ctos_listing_page"] = page
    return result


def fetch_ctos_company_detail(
    url: str,
    *,
    timeout: int = 60,
    max_retries: int = 3,
    retry_sleep_s: float = 2.0,
) -> dict[str, Any]:
    response = _get_with_retries(
        url,
        timeout=timeout,
        max_retries=max_retries,
        retry_sleep_s=retry_sleep_s,
    )
    return parse_ctos_company_detail(response.text)


def fetch_ctos_company_directory(
    *,
    letters: list[str] | None = None,
    pages_per_letter: int = 1,
    fetch_details_limit: int = 0,
    sleep_s: float = 1.0,
    checkpoint_path: str | Path | None = None,
    resume: bool = True,
    checkpoint_every: int = 25,
    timeout: int = 60,
    max_retries: int = 3,
    retry_sleep_s: float = 2.0,
    max_consecutive_failures: int = 5,
) -> pd.DataFrame:
    """Fetch a cautious slice of the public CTOS Malaysia company directory."""
    letters = letters or list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    rows: list[pd.DataFrame] = []
    scanned_pages: set[tuple[str, int]] = set()

    if checkpoint_path is not None and resume:
        cp = Path(checkpoint_path)
        if cp.exists():
            existing = pd.read_parquet(cp)
            if not existing.empty:
                if {"ctos_listing_letter", "ctos_listing_page"}.issubset(existing.columns):
                    rows.append(existing)
                    scanned_pages = set(
                        zip(
                            existing["ctos_listing_letter"].dropna().astype(str),
                            existing["ctos_listing_page"].dropna().astype(int),
                        )
                    )
                    print(f"[INFO] Resuming CTOS crawl from {len(existing):,} cached companies", flush=True)
                else:
                    print("[INFO] Ignoring legacy CTOS cache without page metadata; recrawling snapshot", flush=True)

    for letter in letters:
        consecutive_failures = 0
        for page in range(1, pages_per_letter + 1):
            page_key = (letter.upper(), page)
            if page_key in scanned_pages:
                continue
            try:
                page_df = fetch_ctos_listing_page(
                    letter,
                    page,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_sleep_s=retry_sleep_s,
                )
            except requests.RequestException as exc:
                consecutive_failures += 1
                print(
                    f"[WARN] CTOS {letter.upper()} page {page}: failed after "
                    f"{max_retries} attempts, skipping page ({exc})",
                    flush=True,
                )
                if checkpoint_path is not None and rows:
                    _save_ctos_checkpoint(rows, checkpoint_path)
                if consecutive_failures >= max(max_consecutive_failures, 1):
                    print(
                        f"[WARN] CTOS {letter.upper()}: stopping after "
                        f"{consecutive_failures} consecutive failed pages",
                        flush=True,
                    )
                    break
                time.sleep(sleep_s)
                continue
            if page_df.empty:
                print(f"[INFO] CTOS {letter.upper()} page {page}: empty, stopping letter", flush=True)
                break
            consecutive_failures = 0
            print(f"[INFO] CTOS {letter.upper()} page {page}: {len(page_df)} companies", flush=True)
            rows.append(page_df)
            scanned_pages.add(page_key)
            if checkpoint_path is not None and len(scanned_pages) % max(checkpoint_every, 1) == 0:
                _save_ctos_checkpoint(rows, checkpoint_path)
            time.sleep(sleep_s)

    if not rows:
        return pd.DataFrame(columns=CTOS_COLUMNS)

    if checkpoint_path is not None:
        _save_ctos_checkpoint(rows, checkpoint_path)

    companies = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["ctos_name", "ctos_url"])
    for col in CTOS_COLUMNS:
        if col not in companies.columns:
            companies[col] = None

    if fetch_details_limit > 0:
        detail_rows = []
        for _, row in companies.head(fetch_details_limit).iterrows():
            try:
                detail = fetch_ctos_company_detail(
                    row["ctos_url"],
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_sleep_s=retry_sleep_s,
                )
            except requests.RequestException as exc:
                print(f"[WARN] CTOS detail failed for {row['ctos_url']}: {exc}", flush=True)
                continue
            detail["ctos_url"] = row["ctos_url"]
            detail_rows.append(detail)
            time.sleep(sleep_s)
        details = pd.DataFrame(detail_rows)
        if not details.empty:
            companies = companies.merge(details, on="ctos_url", how="left", suffixes=("", "_detail"))
            for col in [
                "ctos_registration_no", "ctos_nature_of_business",
                "ctos_registration_date", "ctos_state",
            ]:
                detail_col = f"{col}_detail"
                if detail_col in companies.columns:
                    companies[col] = companies[col].combine_first(companies[detail_col])
                    companies.drop(columns=[detail_col], inplace=True)

    return companies[CTOS_COLUMNS].reset_index(drop=True)


def _save_ctos_checkpoint(rows: list[pd.DataFrame], path: str | Path) -> None:
    checkpoint = Path(path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["ctos_name", "ctos_url"],
        keep="first",
    )
    for col in CTOS_COLUMNS:
        if col not in combined.columns:
            combined[col] = None
    combined[CTOS_COLUMNS].to_parquet(checkpoint, index=False)


def match_entities_to_ctos(
    entities: pd.DataFrame,
    ctos_companies: pd.DataFrame,
    *,
    threshold: int = 95,
    fuzzy: bool = True,
) -> pd.DataFrame:
    """Match entity legal names to CTOS company names.

    A match is an incorporation/registration signal for Malaysia, not an
    ownership or UIE signal.
    """
    if entities.empty or ctos_companies.empty:
        return pd.DataFrame(columns=CTOS_MATCH_COLUMNS)

    ctos = ctos_companies.copy()
    ctos["ctos_name_norm"] = ctos["ctos_name"].map(normalize_malaysia_company_name)
    ctos = ctos.dropna(subset=["ctos_name_norm"]).drop_duplicates(subset=["ctos_name_norm"])
    ctos["ctos_first_distinctive_token"] = ctos["ctos_name_norm"].map(
        lambda x: (_distinctive_token_list(x) or [None])[0]
    )
    choices_by_first_token = {
        token: group["ctos_name_norm"].tolist()
        for token, group in ctos.dropna(subset=["ctos_first_distinctive_token"]).groupby(
            "ctos_first_distinctive_token"
        )
    }
    ctos_by_norm = ctos.set_index("ctos_name_norm", drop=False)

    matches: list[dict[str, Any]] = []
    for _, row in entities.iterrows():
        lei = row.get("lei")
        legal_name = row.get("legal_name")
        norm = normalize_malaysia_company_name(legal_name)
        if not norm:
            continue

        method = "fuzzy"
        score = 0
        matched_norm = None
        if norm in ctos_by_norm.index:
            method = "exact"
            score = 100
            matched_norm = norm
        elif fuzzy:
            first_token = (_distinctive_token_list(norm) or [None])[0]
            choices = choices_by_first_token.get(first_token, [])
            result = process.extractOne(norm, choices, scorer=fuzz.token_sort_ratio)
            if result:
                matched_norm, score, _ = result

        if matched_norm is None or score < threshold:
            continue
        if method == "fuzzy" and not _is_safe_fuzzy_ctos_match(norm, matched_norm):
            continue

        ctos_row = ctos_by_norm.loc[matched_norm]
        matches.append({
            "lei": lei,
            "legal_name": legal_name,
            "ctos_name": ctos_row.get("ctos_name"),
            "ctos_url": ctos_row.get("ctos_url"),
            "ctos_registration_no": ctos_row.get("ctos_registration_no"),
            "ctos_match_score": round(float(score), 4),
            "ctos_match_method": method,
            "ctos_registered_malaysia": 1,
        })

    if not matches:
        return pd.DataFrame(columns=CTOS_MATCH_COLUMNS)
    return pd.DataFrame(matches)[CTOS_MATCH_COLUMNS].sort_values(
        ["ctos_match_score", "legal_name"], ascending=[False, True]
    ).reset_index(drop=True)
