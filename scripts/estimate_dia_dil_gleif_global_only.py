from __future__ import annotations

import argparse
import csv
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from rapidfuzz import fuzz

try:
    from lxml import etree

    HAVE_LXML = True
except ImportError:
    import xml.etree.ElementTree as etree

    HAVE_LXML = False

from estimate_dia_dil_gleif_only import (
    UNAVAILABLE_BENCHMARKS,
    clean_scalar,
    normalize_name,
)
from estimate_dia_dil_spreadsheet_uie import workbook_rows, write_xlsx


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "directional_uie"
DEFAULT_INPUT = Path.home() / "Downloads" / "DIA and DIL Entity Name_by Country.xlsx"
DEFAULT_LEI_ZIP = ROOT / "data" / "raw" / "gleif_global_2026-07-15" / "gleif_lei2_20260715.zip"
DEFAULT_RR_ZIP = ROOT / "data" / "raw" / "gleif_global_2026-07-15" / "gleif_rr_20260715.zip"

LEI_NS = "http://www.gleif.org/data/schema/leidata/2016"
RR_NS = "http://www.gleif.org/data/schema/rr/2016"
LEI = f"{{{LEI_NS}}}"
RR = f"{{{RR_NS}}}"

COUNTRY_ALIASES = {
    "UK": "GB",
}

FUZZY_STOPWORDS = {
    "AND",
    "BANK",
    "BHD",
    "CAPITAL",
    "CHEMICAL",
    "CHEMICALS",
    "CO",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "DEVELOPMENT",
    "ELECTRONICS",
    "ENERGY",
    "ENGINEERING",
    "ENTERPRISE",
    "ENTERPRISES",
    "FINANCE",
    "FUND",
    "GLOBAL",
    "GROUP",
    "HOLDING",
    "HOLDINGS",
    "INC",
    "INDUSTRIES",
    "INDUSTRIAL",
    "INTERNATIONAL",
    "INVESTMENT",
    "INVESTMENTS",
    "LIMITED",
    "LLC",
    "LOGISTICS",
    "LTD",
    "MANAGEMENT",
    "MANUFACTURING",
    "MARINE",
    "NATIONAL",
    "PACIFIC",
    "PARTNERS",
    "PLC",
    "PRIVATE",
    "PRODUCTS",
    "PROPERTY",
    "PTE",
    "PUBLIC",
    "RESOURCES",
    "SDN",
    "SERVICES",
    "SOLUTIONS",
    "SYSTEMS",
    "TECHNOLOGY",
    "THE",
    "TRADING",
    "TRUST",
    "VENTURES",
}


@dataclass(frozen=True)
class Query:
    normalized_name: str
    immediate_country: str
    display_name: str

    @property
    def key(self) -> tuple[str, str]:
        return self.normalized_name, self.immediate_country


@dataclass(frozen=True)
class Entity:
    lei: str
    legal_name: str
    country_legal: str
    country_hq: str
    entity_status: str

    @property
    def countries(self) -> set[str]:
        return {value for value in (self.country_legal, self.country_hq) if value}


@dataclass(frozen=True)
class Match:
    entity: Entity | None
    strategy: str
    score: float
    margin: float
    matched_alias: str
    note: str


def country_code(value: Any) -> str:
    code = clean_scalar(value).upper()
    return COUNTRY_ALIASES.get(code, code)


def significant_tokens(value: str) -> set[str]:
    return {
        token
        for token in value.split()
        if len(token) >= 4 and token not in FUZZY_STOPWORDS and not token.isdigit()
    }


def first_zip_member(path: Path):
    archive = zipfile.ZipFile(path)
    members = [member for member in archive.infolist() if not member.is_dir()]
    if len(members) != 1:
        archive.close()
        raise RuntimeError(f"Expected one data file in {path}, found {len(members)}")
    return archive, archive.open(members[0])


def clear_element(element: Any) -> None:
    element.clear()
    if HAVE_LXML:
        parent = element.getparent()
        if parent is not None:
            while element.getprevious() is not None:
                del parent[0]


def iter_xml_records(stream: Any, record_tag: str):
    if HAVE_LXML:
        context = etree.iterparse(
            stream,
            events=("end",),
            tag=record_tag,
            huge_tree=True,
            recover=True,
        )
        for _, element in context:
            yield element
        return

    context = etree.iterparse(stream, events=("start", "end"))
    _, root = next(context)
    for event, element in context:
        if event == "end" and element.tag == record_tag:
            yield element
            element.clear()
            root.clear()


def text_at(element: Any | None, path: str) -> str:
    if element is None:
        return ""
    node = element.find(path)
    return clean_scalar(node.text if node is not None else "")


def extract_entity(record: Any) -> tuple[Entity | None, list[str]]:
    lei = text_at(record, f"{LEI}LEI")
    entity_node = record.find(f"{LEI}Entity")
    legal_name = text_at(entity_node, f"{LEI}LegalName")
    if not lei or not legal_name or entity_node is None:
        return None, []

    names = [legal_name]
    names.extend(
        clean_scalar(node.text)
        for node in entity_node.findall(f"{LEI}OtherEntityNames/{LEI}OtherEntityName")
        if clean_scalar(node.text)
    )
    names.extend(
        clean_scalar(node.text)
        for node in entity_node.findall(
            f"{LEI}TransliteratedOtherEntityNames/{LEI}TransliteratedOtherEntityName"
        )
        if clean_scalar(node.text)
    )
    entity = Entity(
        lei=lei,
        legal_name=legal_name,
        country_legal=country_code(text_at(entity_node, f"{LEI}LegalAddress/{LEI}Country")),
        country_hq=country_code(text_at(entity_node, f"{LEI}HeadquartersAddress/{LEI}Country")),
        entity_status=text_at(entity_node, f"{LEI}EntityStatus").upper(),
    )
    return entity, names


def build_queries(flow_rows: dict[str, list[dict[str, str]]]) -> dict[tuple[str, str], Query]:
    queries: dict[tuple[str, str], Query] = {}
    for rows in flow_rows.values():
        for row in rows:
            display_name = clean_scalar(row.get("NR Counter Party Name"))
            normalized = normalize_name(display_name)
            immediate = country_code(row.get("NR Counter Party Immediate Country Code"))
            if not normalized:
                continue
            query = Query(normalized, immediate, display_name)
            queries.setdefault(query.key, query)
    return queries


def scan_entities_for_matches(
    lei_zip: Path,
    queries: dict[tuple[str, str], Query],
    *,
    progress_every: int,
    exact_only: bool,
    wanted_parent_leis: set[str],
) -> tuple[
    dict[tuple[str, str], dict[str, tuple[Entity, str]]],
    dict[tuple[str, str], dict[str, tuple[Entity, str, float, float, float]]],
    dict[str, Entity],
]:
    exact_by_name: dict[str, set[tuple[str, str]]] = defaultdict(set)
    token_queries: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for key, query in queries.items():
        exact_by_name[query.normalized_name].add(key)
        if not exact_only and query.immediate_country not in {"", "UNK", "ZZ", "OT", "AN"}:
            for token in significant_tokens(query.normalized_name):
                token_queries[(query.immediate_country, token)].add(key)

    token_blocks: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    if not exact_only:
        for key, query in queries.items():
            ranked_tokens = sorted(
                significant_tokens(query.normalized_name),
                key=lambda token: (
                    len(token_queries[(query.immediate_country, token)]),
                    -len(token),
                    token,
                ),
            )
            for token in ranked_tokens[:2]:
                token_blocks[(query.immediate_country, token)].add(key)

    exact: dict[tuple[str, str], dict[str, tuple[Entity, str]]] = defaultdict(dict)
    fuzzy: dict[
        tuple[str, str], dict[str, tuple[Entity, str, float, float, float]]
    ] = defaultdict(dict)
    parent_entities: dict[str, Entity] = {}

    archive, stream = first_zip_member(lei_zip)
    try:
        for count, record in enumerate(
            iter_xml_records(stream, f"{LEI}LEIRecord"), start=1
        ):
            entity, raw_names = extract_entity(record)
            if entity is not None:
                if entity.lei in wanted_parent_leis:
                    parent_entities[entity.lei] = entity
                aliases = {normalize_name(name): name for name in raw_names if normalize_name(name)}
                candidate_queries: set[tuple[str, str]] = set()
                for normalized_alias, raw_alias in aliases.items():
                    for key in exact_by_name.get(normalized_alias, ()):
                        exact[key][entity.lei] = (entity, raw_alias)
                    if not exact_only:
                        for entity_country in entity.countries:
                            for token in significant_tokens(normalized_alias):
                                candidate_queries.update(token_blocks.get((entity_country, token), ()))

                for key in candidate_queries:
                    query = queries[key]
                    best: tuple[str, float, float, float] | None = None
                    for normalized_alias, raw_alias in aliases.items():
                        score = float(fuzz.WRatio(query.normalized_name, normalized_alias))
                        if best is not None and score <= best[1]:
                            continue
                        best = (
                            raw_alias,
                            score,
                            float(fuzz.token_set_ratio(query.normalized_name, normalized_alias)),
                            float(fuzz.ratio(query.normalized_name, normalized_alias)),
                        )
                    if best is None:
                        continue
                    bucket = fuzzy[key]
                    existing = bucket.get(entity.lei)
                    if existing is None or best[1] > existing[2]:
                        if existing is not None or len(bucket) < 5:
                            bucket[entity.lei] = (entity, *best)
                        else:
                            worst_lei, worst = min(bucket.items(), key=lambda item: item[1][2])
                            if best[1] > worst[2]:
                                del bucket[worst_lei]
                                bucket[entity.lei] = (entity, *best)

            clear_element(record)
            if progress_every and count % progress_every == 0:
                print(
                    f"Entity scan: {count:,} records; exact query keys={len(exact):,}; "
                    f"fuzzy query keys={len(fuzzy):,}; parents found={len(parent_entities):,}",
                    flush=True,
                )
    finally:
        stream.close()
        archive.close()
    return exact, fuzzy, parent_entities


def choose_matches(
    queries: dict[tuple[str, str], Query],
    exact: dict[tuple[str, str], dict[str, tuple[Entity, str]]],
    fuzzy: dict[tuple[str, str], dict[str, tuple[Entity, str, float, float, float]]],
    *,
    fuzzy_threshold: float,
    fuzzy_margin: float,
) -> dict[tuple[str, str], Match]:
    matches: dict[tuple[str, str], Match] = {}
    for key, query in queries.items():
        exact_candidates = list(exact.get(key, {}).values())
        if exact_candidates:
            active = [item for item in exact_candidates if item[0].entity_status == "ACTIVE"]
            pool = active or exact_candidates
            country_aligned = [item for item in pool if query.immediate_country in item[0].countries]
            if len(country_aligned) == 1:
                entity, alias = country_aligned[0]
                matches[key] = Match(
                    entity,
                    "gleif_exact_name_country_disambiguated",
                    100.0,
                    100.0,
                    alias,
                    "exact normalised name; immediate country selected one GLEIF entity",
                )
                continue
            if len(pool) == 1:
                entity, alias = pool[0]
                matches[key] = Match(
                    entity,
                    "gleif_unique_exact_name",
                    100.0,
                    100.0,
                    alias,
                    "unique exact normalised GLEIF name",
                )
                continue
            matches[key] = Match(
                None,
                "gleif_ambiguous_exact_name",
                100.0,
                0.0,
                "",
                f"exact name maps to {len(pool)} active/preferred GLEIF entities",
            )
            continue

        ranked = sorted(fuzzy.get(key, {}).values(), key=lambda item: (-item[2], item[0].lei))
        if not ranked:
            matches[key] = Match(None, "gleif_name_unmatched", 0.0, 0.0, "", "no GLEIF candidate")
            continue
        best = ranked[0]
        second_score = ranked[1][2] if len(ranked) > 1 else 0.0
        margin = best[2] - second_score
        entity, alias, score, token_score, char_score = best
        if (
            score >= fuzzy_threshold
            and margin >= fuzzy_margin
            and token_score >= 95.0
            and char_score >= 85.0
            and query.immediate_country in entity.countries
        ):
            matches[key] = Match(
                entity,
                "gleif_high_confidence_fuzzy_same_country",
                score,
                margin,
                alias,
                f"token_set={token_score:.1f}; char_ratio={char_score:.1f}; same-country guard passed",
            )
        else:
            matches[key] = Match(
                None,
                "gleif_fuzzy_candidate_rejected",
                score,
                margin,
                alias,
                f"best candidate failed conservative guards; token_set={token_score:.1f}; char_ratio={char_score:.1f}",
            )
    return matches


def scan_relationships(
    rr_zip: Path,
    source_leis: set[str] | None,
    *,
    progress_every: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    ultimate: dict[str, set[str]] = defaultdict(set)
    direct: dict[str, set[str]] = defaultdict(set)
    archive, stream = first_zip_member(rr_zip)
    try:
        for count, record in enumerate(
            iter_xml_records(stream, f"{RR}RelationshipRecord"), start=1
        ):
            relationship = record.find(f"{RR}Relationship")
            source = text_at(relationship, f"{RR}StartNode/{RR}NodeID")
            if source_leis is None or source in source_leis:
                target = text_at(relationship, f"{RR}EndNode/{RR}NodeID")
                kind = text_at(relationship, f"{RR}RelationshipType").upper()
                status = text_at(relationship, f"{RR}RelationshipStatus").upper()
                if target and status in {"", "ACTIVE"}:
                    if kind == "IS_ULTIMATELY_CONSOLIDATED_BY":
                        ultimate[source].add(target)
                    elif kind == "IS_DIRECTLY_CONSOLIDATED_BY":
                        direct[source].add(target)
            clear_element(record)
            if progress_every and count % progress_every == 0:
                print(f"Relationship scan: {count:,} records", flush=True)
    finally:
        stream.close()
        archive.close()
    return ultimate, direct


def scan_entities_by_lei(
    lei_zip: Path,
    wanted_leis: set[str],
    *,
    progress_every: int,
) -> dict[str, Entity]:
    found: dict[str, Entity] = {}
    if not wanted_leis:
        return found
    archive, stream = first_zip_member(lei_zip)
    try:
        for count, record in enumerate(
            iter_xml_records(stream, f"{LEI}LEIRecord"), start=1
        ):
            entity, _ = extract_entity(record)
            if entity is not None and entity.lei in wanted_leis:
                found[entity.lei] = entity
                if len(found) == len(wanted_leis):
                    clear_element(record)
                    break
            clear_element(record)
            if progress_every and count % progress_every == 0:
                print(
                    f"Parent detail scan: {count:,} records; found={len(found):,}/{len(wanted_leis):,}",
                    flush=True,
                )
    finally:
        stream.close()
        archive.close()
    return found


def one_target(values: Iterable[str]) -> str:
    unique = sorted({value for value in values if value})
    return unique[0] if len(unique) == 1 else ""


def process_flow(
    flow: str,
    rows: list[dict[str, str]],
    matches: dict[tuple[str, str], Match],
    ultimate: dict[str, set[str]],
    direct: dict[str, set[str]],
    parent_entities: dict[str, Entity],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in rows:
        counterparty = clean_scalar(row.get("NR Counter Party Name"))
        key = (
            normalize_name(counterparty),
            country_code(row.get("NR Counter Party Immediate Country Code")),
        )
        match = matches.get(key, Match(None, "gleif_name_unmatched", 0.0, 0.0, "", "blank/no query"))
        source = match.entity
        source_lei = source.lei if source else ""
        ultimate_targets = ultimate.get(source_lei, set())
        direct_targets = direct.get(source_lei, set())
        ultimate_lei = one_target(ultimate_targets)
        direct_lei = one_target(direct_targets)
        ultimate_entity = parent_entities.get(ultimate_lei)
        direct_entity = parent_entities.get(direct_lei)
        ultimate_country = ""
        if ultimate_entity is not None:
            ultimate_country = ultimate_entity.country_legal or ultimate_entity.country_hq

        if source is None:
            status = "gleif_counterparty_unmatched"
        elif len(ultimate_targets) > 1:
            status = "gleif_multiple_reported_ultimate_parents"
        elif ultimate_lei and ultimate_entity is None:
            status = "gleif_ultimate_parent_record_unavailable"
        elif ultimate_lei and not ultimate_country:
            status = "gleif_ultimate_parent_country_unavailable"
        elif ultimate_country:
            status = "uie_from_gleif_reported_ultimate_parent"
        elif direct_lei:
            status = "gleif_counterparty_matched_direct_parent_only"
        else:
            status = "gleif_counterparty_matched_no_reported_parent"

        output.append(
            {
                "flow_type": flow,
                "source_excel_row": clean_scalar(row.get("source_excel_row")),
                "resident_entity_name": clean_scalar(row.get("RE Name 2 filled")),
                "nonresident_counterparty_name": counterparty,
                "immediate_country": clean_scalar(row.get("NR Counter Party Immediate Country Code")),
                "survey_uie_country_reference_only": clean_scalar(
                    row.get("NR Counter Party Ultimate Country Code")
                ),
                "gleif_only_estimated_uie_country": ultimate_country,
                "gleif_only_estimated_uie_name": ultimate_entity.legal_name if ultimate_entity else "",
                "gleif_only_estimated_uie_lei": ultimate_lei,
                "result_status": status,
                "counterparty_match_strategy": match.strategy,
                "counterparty_match_score": f"{match.score:.1f}" if match.score else "",
                "counterparty_match_margin": f"{match.margin:.1f}" if match.score else "",
                "matched_gleif_alias": match.matched_alias,
                "matched_counterparty_lei": source_lei,
                "matched_gleif_legal_name": source.legal_name if source else "",
                "matched_gleif_entity_status": source.entity_status if source else "",
                "matched_gleif_legal_country": source.country_legal if source else "",
                "matched_gleif_hq_country": source.country_hq if source else "",
                "gleif_direct_parent_country": (
                    direct_entity.country_legal or direct_entity.country_hq if direct_entity else ""
                ),
                "gleif_direct_parent_name": direct_entity.legal_name if direct_entity else "",
                "gleif_direct_parent_lei": direct_lei,
                "match_note": match.note,
                "method_note": (
                    "GLEIF-only global counterparty match plus GLEIF-reported ultimate-parent relationship. "
                    "Immediate country is used only to disambiguate names. No curated aliases, address "
                    "propagation, broader graph inference, ML, default-immediate assignment, or survey UIE "
                    "is used to produce the estimate."
                ),
            }
        )
    return output


def score_rows(rows: list[dict[str, str]]) -> dict[str, int | float]:
    matched = [row for row in rows if row["matched_counterparty_lei"]]
    assigned = [row for row in rows if row["gleif_only_estimated_uie_country"]]
    scorable = [
        row
        for row in assigned
        if row["survey_uie_country_reference_only"].upper() not in UNAVAILABLE_BENCHMARKS
    ]
    hits = sum(
        row["gleif_only_estimated_uie_country"].upper()
        == row["survey_uie_country_reference_only"].upper()
        for row in scorable
    )
    return {
        "rows": len(rows),
        "unique_counterparty_names": len(
            {
                normalize_name(row["nonresident_counterparty_name"])
                for row in rows
                if normalize_name(row["nonresident_counterparty_name"])
            }
        ),
        "entity_matched": len(matched),
        "entity_match_coverage": len(matched) / len(rows) if rows else 0.0,
        "uie_assigned": len(assigned),
        "uie_assignment_coverage": len(assigned) / len(rows) if rows else 0.0,
        "scorable_assigned": len(scorable),
        "hits": hits,
        "conditional_hit_rate": hits / len(scorable) if scorable else 0.0,
        "exact_matches": sum(
            bool(row["matched_counterparty_lei"])
            and "exact" in row["counterparty_match_strategy"]
            for row in rows
        ),
        "fuzzy_matches": sum(
            row["counterparty_match_strategy"] == "gleif_high_confidence_fuzzy_same_country"
            for row in rows
        ),
    }


def summary_rows(flow_scores: dict[str, dict[str, int | float]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for flow, metrics in flow_scores.items():
        output.append(
            {
                "flow": flow,
                "rows": f'{int(metrics["rows"]):,}',
                "unique_counterparty_names": f'{int(metrics["unique_counterparty_names"]):,}',
                "counterparty_matched_rows": f'{int(metrics["entity_matched"]):,}',
                "counterparty_match_coverage": f'{float(metrics["entity_match_coverage"]):.2%}',
                "gleif_ultimate_uie_rows": f'{int(metrics["uie_assigned"]):,}',
                "uie_assignment_coverage": f'{float(metrics["uie_assignment_coverage"]):.2%}',
                "scorable_assigned_rows": f'{int(metrics["scorable_assigned"]):,}',
                "exact_country_hits": f'{int(metrics["hits"]):,}',
                "conditional_hit_rate": f'{float(metrics["conditional_hit_rate"]):.2%}',
                "exact_entity_matches": f'{int(metrics["exact_matches"]):,}',
                "fuzzy_entity_matches": f'{int(metrics["fuzzy_matches"]):,}',
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate DIA/DIL UIE from global GLEIF names and reported ultimate-parent data only."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--lei-zip", type=Path, default=DEFAULT_LEI_ZIP)
    parser.add_argument("--rr-zip", type=Path, default=DEFAULT_RR_ZIP)
    parser.add_argument("--stamp", default=date.today().isoformat())
    parser.add_argument("--fuzzy-threshold", type=float, default=96.0)
    parser.add_argument("--fuzzy-margin", type=float, default=3.0)
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--exact-only", action="store_true")
    args = parser.parse_args()

    sheets = workbook_rows(args.input)
    flow_rows = {
        "DIL": sheets.get("Details_Inward FDI_EQ_Country", []),
        "DIA": sheets.get("Details_Outward FDI_EQ_Country", []),
    }
    if not all(flow_rows.values()):
        raise RuntimeError(f"Expected inward and outward detail sheets in {args.input}")

    queries = build_queries(flow_rows)
    print(f"Unique counterparty/country queries: {len(queries):,}", flush=True)
    all_ultimate, all_direct = scan_relationships(
        args.rr_zip,
        None,
        progress_every=args.progress_every,
    )
    parent_leis = {
        target
        for mapping in (all_ultimate, all_direct)
        for targets in mapping.values()
        for target in targets
        if target
    }
    print(f"GLEIF relationship targets to retain: {len(parent_leis):,}", flush=True)
    exact, fuzzy, parent_entities = scan_entities_for_matches(
        args.lei_zip,
        queries,
        progress_every=args.progress_every,
        exact_only=args.exact_only,
        wanted_parent_leis=parent_leis,
    )
    matches = choose_matches(
        queries,
        exact,
        fuzzy,
        fuzzy_threshold=args.fuzzy_threshold,
        fuzzy_margin=args.fuzzy_margin,
    )
    source_leis = {match.entity.lei for match in matches.values() if match.entity is not None}
    print(f"Accepted GLEIF counterparty entities: {len(source_leis):,}", flush=True)
    ultimate = {source: all_ultimate[source] for source in source_leis if source in all_ultimate}
    direct = {source: all_direct[source] for source in source_leis if source in all_direct}

    output_by_flow = {
        flow: process_flow(flow, rows, matches, ultimate, direct, parent_entities)
        for flow, rows in flow_rows.items()
    }
    scores = {flow: score_rows(rows) for flow, rows in output_by_flow.items()}
    summary = summary_rows(scores)

    stem = f"dia_dil_spreadsheet_gleif_global_counterparty_only_{args.stamp}"
    workbook = REPORTS / f"{stem}.xlsx"
    summary_csv = REPORTS / f"{stem}_summary.csv"
    note = REPORTS / f"{stem}.md"
    write_csv(summary_csv, summary)
    for flow, rows in output_by_flow.items():
        write_csv(REPORTS / f"{stem}_{flow.lower()}.csv", rows)
    write_xlsx(
        workbook,
        {
            "summary": summary,
            "DIL_GLEIF_global": output_by_flow["DIL"],
            "DIA_GLEIF_global": output_by_flow["DIA"],
        },
    )

    note.write_text(
        "\n".join(
            [
                "# DIA/DIL Global GLEIF-Only Counterparty UIE Experiment",
                "",
                f"Generated: {args.stamp}",
                "GLEIF bulk snapshot: 2026-07-15",
                f"Input: `{args.input.name}`",
                "",
                "## Method",
                "",
                "Each non-resident counterparty name is matched locally against the global GLEIF Level 1 bulk file. Unique exact normalised names are preferred. The spreadsheet immediate-country code is used only to disambiguate duplicate exact names and to constrain conservative fuzzy candidates.",
                "",
                "A UIE is assigned only when the matched counterparty LEI has an active `IS_ULTIMATELY_CONSOLIDATED_BY` relationship in GLEIF Level 2 and the reported parent LEI has a usable country. Direct-parent links are diagnostic only. No curated group aliases, address propagation, broader graph inference, machine learning, default-immediate assignment or survey UIE is used to produce the estimate.",
                "",
                "The survey UIE field is copied only for after-the-fact validation. UNK, ZZ and blank reference values are excluded from exact-country accuracy.",
                "",
                "## Results",
                "",
                "| Flow | Rows | Counterparty match coverage | GLEIF ultimate UIE coverage | Scorable assigned | Hits | Conditional hit rate |",
                "|---|---:|---:|---:|---:|---:|---:|",
                *[
                    f'| {row["flow"]} | {row["rows"]} | {row["counterparty_match_coverage"]} | {row["uie_assignment_coverage"]} | {row["scorable_assigned_rows"]} | {row["exact_country_hits"]} | {row["conditional_hit_rate"]} |'
                    for row in summary
                ],
                "",
                "## Interpretation",
                "",
                "This is the clean GLEIF-only baseline for the spreadsheet counterparties. It measures what can be recovered from public GLEIF entity names and explicitly reported ultimate-parent relationships before any broader project methodology is added.",
            ]
        ),
        encoding="utf-8",
    )

    for row in summary:
        print(row, flush=True)
    print(workbook, flush=True)


if __name__ == "__main__":
    main()
