from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

from estimate_dia_dil_spreadsheet_uie import workbook_rows, write_xlsx


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "directional_uie"
DEFAULT_INPUT = Path.home() / "Downloads" / "DIA and DIL Entity Name_by Country.xlsx"
MY_ENTITIES = ROOT / "data" / "raw" / "gleif_my_lei.parquet"
MY_RELATED = ROOT / "data" / "raw" / "gleif_my_related_lei.parquet"
MY_RELATIONSHIPS = ROOT / "data" / "interim" / "gleif_my_relationships.parquet"

UNAVAILABLE_BENCHMARKS = {"", "UNK", "ZZ"}
LEGAL_CANONICALISATION = (
    (re.compile(r"\bSENDIRIAN\s+BERHAD\b"), "SDN BHD"),
    (re.compile(r"\bSDN\.?\s*BHD\.?\b"), "SDN BHD"),
    (re.compile(r"\bPUBLIC\s+LIMITED\s+COMPANY\b"), "PLC"),
    (re.compile(r"\bPTE\.?\s*LTD\.?\b"), "PTE LTD"),
    (re.compile(r"\bLIMITED\b"), "LTD"),
    (re.compile(r"\bBERHAD\b"), "BHD"),
)


@dataclass(frozen=True)
class EntityCandidate:
    lei: str
    legal_name: str
    country_legal: str
    country_hq: str
    entity_status: str


@dataclass(frozen=True)
class NameMatch:
    candidate: EntityCandidate | None
    strategy: str
    score: float
    margin: float
    matched_alias: str
    note: str


def clean_scalar(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean_scalar(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"^[\"']+|[\"']+$", "", text)
    text = re.sub(r"\([^)]{1,12}\)", " ", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    for pattern, replacement in LEGAL_CANONICALISATION:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def iter_other_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = clean_scalar(item.get("name"))
        else:
            name = clean_scalar(item)
        if name:
            names.append(name)
    return names


def load_gleif_entities() -> tuple[dict[str, EntityCandidate], dict[str, list[str]], list[str]]:
    entities = pd.read_parquet(MY_ENTITIES)
    by_lei: dict[str, EntityCandidate] = {}
    alias_to_leis: dict[str, set[str]] = {}

    for _, row in entities.iterrows():
        lei = clean_scalar(row.get("lei"))
        legal_name = clean_scalar(row.get("legal_name"))
        if not lei or not legal_name:
            continue
        candidate = EntityCandidate(
            lei=lei,
            legal_name=legal_name,
            country_legal=clean_scalar(row.get("country_legal")),
            country_hq=clean_scalar(row.get("country_hq")),
            entity_status=clean_scalar(row.get("entity_status")),
        )
        by_lei[lei] = candidate
        aliases = [legal_name]
        aliases.extend(iter_other_names(row.get("other_names")))
        aliases.extend(iter_other_names(row.get("transliterated_other_names")))
        for alias in aliases:
            norm = normalize_name(alias)
            if norm:
                alias_to_leis.setdefault(norm, set()).add(lei)

    stable_aliases = {alias: sorted(leis) for alias, leis in alias_to_leis.items()}
    return by_lei, stable_aliases, sorted(stable_aliases)


def match_entity_name(
    name: str,
    by_lei: dict[str, EntityCandidate],
    alias_to_leis: dict[str, list[str]],
    aliases: list[str],
    *,
    fuzzy_threshold: float,
    fuzzy_margin: float,
    exact_only: bool,
) -> NameMatch:
    query = normalize_name(name)
    if not query:
        return NameMatch(None, "unmatched_blank_name", 0.0, 0.0, "", "blank resident entity name")

    exact_leis = alias_to_leis.get(query, [])
    if len(exact_leis) == 1:
        return NameMatch(by_lei[exact_leis[0]], "gleif_exact_name", 100.0, 100.0, query, "unique exact normalised GLEIF name")
    if len(exact_leis) > 1:
        return NameMatch(None, "ambiguous_exact_name", 100.0, 0.0, query, "exact name maps to multiple LEIs")
    if exact_only or len(query) < 8:
        return NameMatch(None, "unmatched_no_exact_name", 0.0, 0.0, "", "no exact GLEIF name match")

    top = process.extract(query, aliases, scorer=fuzz.WRatio, limit=3)
    if not top:
        return NameMatch(None, "unmatched_no_candidate", 0.0, 0.0, "", "GLEIF alias index returned no candidate")
    best_alias, best_score, _ = top[0]
    second_score = top[1][1] if len(top) > 1 else 0.0
    margin = float(best_score - second_score)
    token_score = float(fuzz.token_set_ratio(query, best_alias))
    char_score = float(fuzz.ratio(query, best_alias))
    leis = alias_to_leis.get(best_alias, [])

    if len(leis) != 1:
        return NameMatch(None, "ambiguous_fuzzy_name", float(best_score), margin, best_alias, "best fuzzy alias maps to multiple LEIs")
    if best_score < fuzzy_threshold:
        return NameMatch(None, "unmatched_fuzzy_below_threshold", float(best_score), margin, best_alias, f"score below {fuzzy_threshold:.1f}")
    if margin < fuzzy_margin:
        return NameMatch(None, "ambiguous_fuzzy_margin", float(best_score), margin, best_alias, f"best/second margin below {fuzzy_margin:.1f}")
    if token_score < 95.0 or char_score < 85.0:
        return NameMatch(None, "unmatched_fuzzy_token_guard", float(best_score), margin, best_alias, "candidate failed token/character similarity guard")
    return NameMatch(
        by_lei[leis[0]],
        "gleif_high_confidence_fuzzy",
        float(best_score),
        margin,
        best_alias,
        f"accepted with token_set={token_score:.1f}; char_ratio={char_score:.1f}",
    )


def load_parent_evidence() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    related = pd.read_parquet(MY_RELATED)
    domestic = pd.read_parquet(MY_ENTITIES)
    parent_records = pd.concat([related, domestic], ignore_index=True).drop_duplicates(subset=["lei"], keep="first")
    parent_by_lei = {
        clean_scalar(row.get("lei")): {
            "lei": clean_scalar(row.get("lei")),
            "name": clean_scalar(row.get("legal_name")),
            "country": clean_scalar(row.get("country_legal")) or clean_scalar(row.get("country_hq")),
        }
        for _, row in parent_records.iterrows()
        if clean_scalar(row.get("lei"))
    }

    relationships = pd.read_parquet(MY_RELATIONSHIPS)
    relationships = relationships[
        relationships["relationship_status"].fillna("").str.upper().isin({"", "ACTIVE"})
    ]

    def relationship_map(kind: str) -> dict[str, dict[str, str]]:
        subset = relationships[relationships["relationship_type"] == kind]
        result: dict[str, dict[str, str]] = {}
        for source, group in subset.groupby("source_lei", dropna=True):
            targets = sorted({clean_scalar(v) for v in group["target_lei"] if clean_scalar(v)})
            if len(targets) != 1:
                result[clean_scalar(source)] = {
                    "lei": "",
                    "name": "",
                    "country": "",
                    "status": "ambiguous_multiple_targets" if targets else "missing_target",
                }
                continue
            target = targets[0]
            parent = parent_by_lei.get(target, {"lei": target, "name": "", "country": ""})
            result[clean_scalar(source)] = {**parent, "status": "reported_active_relationship"}
        return result

    return relationship_map("ultimate_parent"), relationship_map("direct_parent")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def score_rows(rows: list[dict[str, str]]) -> dict[str, int | float]:
    entity_matched = [row for row in rows if row["matched_lei"]]
    assigned = [row for row in rows if row["gleif_only_estimated_uie_country"]]
    scorable = [
        row
        for row in assigned
        if row["survey_uie_country_reference"].upper() not in UNAVAILABLE_BENCHMARKS
    ]
    hits = sum(
        row["gleif_only_estimated_uie_country"].upper()
        == row["survey_uie_country_reference"].upper()
        for row in scorable
    )
    return {
        "rows": len(rows),
        "unique_resident_names": len({normalize_name(row["resident_entity_name"]) for row in rows if normalize_name(row["resident_entity_name"])}),
        "entity_matched": len(entity_matched),
        "entity_match_coverage": len(entity_matched) / len(rows) if rows else 0.0,
        "uie_assigned": len(assigned),
        "uie_assignment_coverage": len(assigned) / len(rows) if rows else 0.0,
        "scorable_assigned": len(scorable),
        "hits": hits,
        "conditional_hit_rate": hits / len(scorable) if scorable else 0.0,
        "exact_matches": sum(row["entity_match_strategy"] == "gleif_exact_name" for row in rows),
        "fuzzy_matches": sum(row["entity_match_strategy"] == "gleif_high_confidence_fuzzy" for row in rows),
        "direct_parent_only": sum(row["result_status"] == "gleif_entity_matched_direct_parent_only" for row in rows),
    }


def process_flow(
    flow: str,
    input_rows: list[dict[str, str]],
    *,
    by_lei: dict[str, EntityCandidate],
    alias_to_leis: dict[str, list[str]],
    aliases: list[str],
    ultimate_by_source: dict[str, dict[str, str]],
    direct_by_source: dict[str, dict[str, str]],
    fuzzy_threshold: float,
    fuzzy_margin: float,
    exact_only: bool,
) -> list[dict[str, str]]:
    cache: dict[str, NameMatch] = {}
    out: list[dict[str, str]] = []
    for row in input_rows:
        resident_name = clean_scalar(row.get("RE Name 2 filled"))
        cache_key = normalize_name(resident_name)
        if cache_key not in cache:
            cache[cache_key] = match_entity_name(
                resident_name,
                by_lei,
                alias_to_leis,
                aliases,
                fuzzy_threshold=fuzzy_threshold,
                fuzzy_margin=fuzzy_margin,
                exact_only=exact_only,
            )
        match = cache[cache_key]
        candidate = match.candidate
        lei = candidate.lei if candidate else ""
        ultimate = ultimate_by_source.get(lei, {}) if lei else {}
        direct = direct_by_source.get(lei, {}) if lei else {}
        ultimate_country = clean_scalar(ultimate.get("country"))

        if not candidate:
            status = "gleif_entity_unmatched"
        elif ultimate_country:
            status = "uie_from_gleif_reported_ultimate_parent"
        elif direct.get("lei"):
            status = "gleif_entity_matched_direct_parent_only"
        elif ultimate.get("lei") and not ultimate_country:
            status = "gleif_ultimate_parent_country_unavailable"
        else:
            status = "gleif_entity_matched_no_parent_relationship"

        out.append(
            {
                "flow_type": flow,
                "source_excel_row": clean_scalar(row.get("source_excel_row")),
                "resident_entity_name": resident_name,
                "nonresident_counterparty_name": clean_scalar(row.get("NR Counter Party Name")),
                "immediate_country": clean_scalar(row.get("NR Counter Party Immediate Country Code")),
                "survey_uie_country_reference": clean_scalar(row.get("NR Counter Party Ultimate Country Code")),
                "gleif_only_estimated_uie_country": ultimate_country,
                "gleif_only_estimated_uie_name": clean_scalar(ultimate.get("name")),
                "gleif_only_estimated_uie_lei": clean_scalar(ultimate.get("lei")),
                "result_status": status,
                "entity_match_strategy": match.strategy,
                "entity_match_score": f"{match.score:.1f}" if match.score else "",
                "entity_match_margin": f"{match.margin:.1f}" if match.score else "",
                "matched_gleif_alias": match.matched_alias,
                "matched_lei": lei,
                "matched_gleif_legal_name": candidate.legal_name if candidate else "",
                "matched_gleif_entity_status": candidate.entity_status if candidate else "",
                "matched_gleif_country": candidate.country_legal if candidate else "",
                "gleif_direct_parent_country": clean_scalar(direct.get("country")),
                "gleif_direct_parent_name": clean_scalar(direct.get("name")),
                "gleif_direct_parent_lei": clean_scalar(direct.get("lei")),
                "match_note": match.note,
                "method_note": "GLEIF-only: resident entity name match plus reported ultimate-parent relationship; no aliases, address propagation, ML, survey-UIE inference, or immediate-country fallback.",
            }
        )
    return out


def summary_rows(dil: dict[str, int | float], dia: dict[str, int | float]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for flow, metrics in (("DIL", dil), ("DIA", dia)):
        rows.append(
            {
                "flow": flow,
                "rows": f'{int(metrics["rows"]):,}',
                "unique_resident_names": f'{int(metrics["unique_resident_names"]):,}',
                "entity_matched_rows": f'{int(metrics["entity_matched"]):,}',
                "entity_match_coverage": f'{float(metrics["entity_match_coverage"]):.2%}',
                "gleif_ultimate_uie_rows": f'{int(metrics["uie_assigned"]):,}',
                "uie_assignment_coverage": f'{float(metrics["uie_assignment_coverage"]):.2%}',
                "scorable_assigned_rows": f'{int(metrics["scorable_assigned"]):,}',
                "exact_country_hits": f'{int(metrics["hits"]):,}',
                "conditional_hit_rate": f'{float(metrics["conditional_hit_rate"]):.2%}',
                "exact_entity_matches": f'{int(metrics["exact_matches"]):,}',
                "fuzzy_entity_matches": f'{int(metrics["fuzzy_matches"]):,}',
                "direct_parent_only_rows": f'{int(metrics["direct_parent_only"]):,}',
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate DIA/DIL UIE from GLEIF entity and reported ultimate-parent data only.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stamp", default=date.today().isoformat())
    parser.add_argument("--fuzzy-threshold", type=float, default=96.0)
    parser.add_argument("--fuzzy-margin", type=float, default=3.0)
    parser.add_argument("--exact-only", action="store_true")
    args = parser.parse_args()

    sheets = workbook_rows(args.input)
    dil_input = sheets.get("Details_Inward FDI_EQ_Country", [])
    dia_input = sheets.get("Details_Outward FDI_EQ_Country", [])
    if not dil_input or not dia_input:
        raise RuntimeError(f"Expected inward and outward detail sheets in {args.input}")

    by_lei, alias_to_leis, aliases = load_gleif_entities()
    ultimate_by_source, direct_by_source = load_parent_evidence()
    common = {
        "by_lei": by_lei,
        "alias_to_leis": alias_to_leis,
        "aliases": aliases,
        "ultimate_by_source": ultimate_by_source,
        "direct_by_source": direct_by_source,
        "fuzzy_threshold": args.fuzzy_threshold,
        "fuzzy_margin": args.fuzzy_margin,
        "exact_only": args.exact_only,
    }
    dil_rows = process_flow("DIL", dil_input, **common)
    dia_rows = process_flow("DIA", dia_input, **common)
    dil_score = score_rows(dil_rows)
    dia_score = score_rows(dia_rows)
    summary = summary_rows(dil_score, dia_score)

    stem = f"dia_dil_spreadsheet_gleif_only_{args.stamp}"
    dil_csv = REPORTS / f"{stem}_dil.csv"
    dia_csv = REPORTS / f"{stem}_dia.csv"
    summary_csv = REPORTS / f"{stem}_summary.csv"
    workbook = REPORTS / f"{stem}.xlsx"
    note = REPORTS / f"{stem}.md"
    write_csv(dil_csv, dil_rows)
    write_csv(dia_csv, dia_rows)
    write_csv(summary_csv, summary)
    write_xlsx(workbook, {"summary": summary, "DIL_GLEIF_only": dil_rows, "DIA_GLEIF_only": dia_rows})

    note.write_text(
        "\n".join(
            [
                "# DIA/DIL GLEIF-Only UIE Experiment",
                "",
                f"Generated: {args.stamp}",
                f"Input: `{args.input.name}`",
                "",
                "## Method",
                "",
                "Each spreadsheet row is linked using only the Malaysian resident entity name against the Malaysia GLEIF LEI snapshot. Unique exact normalised matches are preferred; conservative high-confidence fuzzy matches are accepted only when the score, second-candidate margin, token similarity and character similarity guards all pass.",
                "",
                "A UIE is assigned only when GLEIF reports an active ultimate-parent relationship for the matched resident LEI and the ultimate parent record has a usable country. A direct-parent relationship is reported as a diagnostic but is not treated as UIE. The run does not use curated group aliases, address propagation, graph inference beyond the reported ultimate-parent endpoint, machine learning, immediate-country fallback or the spreadsheet UIE field for assignment.",
                "",
                "The spreadsheet UIE field is copied only for after-the-fact validation. UNK, ZZ and blank reference values are excluded from exact-country accuracy.",
                "",
                "## Results",
                "",
                "| Flow | Rows | Entity match coverage | GLEIF ultimate UIE coverage | Scorable assigned | Hits | Conditional hit rate |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *[
                    f'| {row["flow"]} | {row["rows"]} | {row["entity_match_coverage"]} | {row["uie_assignment_coverage"]} | {row["scorable_assigned_rows"]} | {row["exact_country_hits"]} | {row["conditional_hit_rate"]} |'
                    for row in summary
                ],
                "",
                "## Outputs",
                "",
                f"- `{workbook.name}`",
                f"- `{dil_csv.name}`",
                f"- `{dia_csv.name}`",
                f"- `{summary_csv.name}`",
            ]
        ),
        encoding="utf-8",
    )

    for row in summary:
        print(row)
    print(workbook)


if __name__ == "__main__":
    main()
