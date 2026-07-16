from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date
from pathlib import Path

from estimate_dia_dil_gleif_only import clean_scalar, normalize_name
from estimate_dia_dil_spreadsheet_uie import write_xlsx


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports" / "directional_uie"
DEFAULT_SOURCE_STEM = "dia_dil_spreadsheet_gleif_global_counterparty_only_2026-07-16"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assign_economy(row: dict[str, str]) -> dict[str, str]:
    ultimate_country = clean_scalar(row.get("gleif_only_estimated_uie_country"))
    direct_country = clean_scalar(row.get("gleif_direct_parent_country"))
    legal_country = clean_scalar(row.get("matched_gleif_legal_country"))
    hq_country = clean_scalar(row.get("matched_gleif_hq_country"))

    if ultimate_country:
        economy = ultimate_country
        name = clean_scalar(row.get("gleif_only_estimated_uie_name"))
        lei = clean_scalar(row.get("gleif_only_estimated_uie_lei"))
        level = "reported_ultimate_parent"
        is_uie = "TRUE"
        note = "GLEIF reports an active ultimate consolidated parent."
    elif direct_country:
        economy = direct_country
        name = clean_scalar(row.get("gleif_direct_parent_name"))
        lei = clean_scalar(row.get("gleif_direct_parent_lei"))
        level = "reported_direct_parent_nearest"
        is_uie = "FALSE"
        note = "No reported ultimate parent; nearest available GLEIF parent economy used."
    elif legal_country:
        economy = legal_country
        name = clean_scalar(row.get("matched_gleif_legal_name"))
        lei = clean_scalar(row.get("matched_counterparty_lei"))
        level = "matched_entity_legal_economy"
        is_uie = "FALSE"
        note = "No reported parent economy; matched entity legal economy used."
    elif hq_country:
        economy = hq_country
        name = clean_scalar(row.get("matched_gleif_legal_name"))
        lei = clean_scalar(row.get("matched_counterparty_lei"))
        level = "matched_entity_hq_economy"
        is_uie = "FALSE"
        note = "No reported parent or legal economy; matched entity HQ economy used."
    else:
        economy = ""
        name = ""
        lei = ""
        level = "unresolved_no_gleif_entity"
        is_uie = "FALSE"
        note = "Entity was not safely matched to GLEIF or has no usable GLEIF economy."

    selected = {
        "assigned_economy": economy,
        "assigned_entity_name": name,
        "assigned_lei": lei,
        "assignment_level": level,
        "is_reported_uie": is_uie,
        "assignment_note": note,
    }
    leading = {
        "flow_type": clean_scalar(row.get("flow_type")),
        "source_excel_row": clean_scalar(row.get("source_excel_row")),
        "resident_entity_name": clean_scalar(row.get("resident_entity_name")),
        "nonresident_counterparty_name": clean_scalar(row.get("nonresident_counterparty_name")),
        "immediate_country": clean_scalar(row.get("immediate_country")),
        **selected,
    }
    remainder = {key: value for key, value in row.items() if key not in leading}
    return {**leading, **remainder}


def entity_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            normalize_name(row.get("nonresident_counterparty_name", "")),
            clean_scalar(row.get("immediate_country")).upper(),
        )
        if not key[0]:
            continue
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, str]] = []
    for (_, immediate), members in grouped.items():
        representative = members[0]
        alternatives = {
            (
                member["assigned_economy"],
                member["assignment_level"],
                member["assigned_lei"],
            )
            for member in members
        }
        output.append(
            {
                "flow_type": representative["flow_type"],
                "nonresident_counterparty_name": representative["nonresident_counterparty_name"],
                "immediate_country": immediate,
                "row_occurrences": str(len(members)),
                "assigned_economy": representative["assigned_economy"],
                "assigned_entity_name": representative["assigned_entity_name"],
                "assigned_lei": representative["assigned_lei"],
                "assignment_level": representative["assignment_level"],
                "is_reported_uie": representative["is_reported_uie"],
                "consistent_across_rows": "TRUE" if len(alternatives) == 1 else "FALSE",
                "matched_gleif_legal_name": representative.get("matched_gleif_legal_name", ""),
                "matched_counterparty_lei": representative.get("matched_counterparty_lei", ""),
                "matched_gleif_legal_country": representative.get("matched_gleif_legal_country", ""),
                "matched_gleif_hq_country": representative.get("matched_gleif_hq_country", ""),
                "counterparty_match_strategy": representative.get("counterparty_match_strategy", ""),
                "assignment_note": representative["assignment_note"],
            }
        )
    return sorted(output, key=lambda row: (row["assignment_level"], row["nonresident_counterparty_name"]))


def summary_row(flow: str, rows: list[dict[str, str]], entities: list[dict[str, str]]) -> dict[str, str]:
    row_levels = Counter(row["assignment_level"] for row in rows)
    entity_levels = Counter(row["assignment_level"] for row in entities)
    assigned_rows = sum(bool(row["assigned_economy"]) for row in rows)
    assigned_entities = sum(bool(row["assigned_economy"]) for row in entities)
    return {
        "flow": flow,
        "rows": f"{len(rows):,}",
        "unique_entity_country_keys": f"{len(entities):,}",
        "rows_with_economy": f"{assigned_rows:,}",
        "row_assignment_coverage": f"{assigned_rows / len(rows):.2%}" if rows else "0.00%",
        "entity_country_keys_with_economy": f"{assigned_entities:,}",
        "entity_country_key_coverage": f"{assigned_entities / len(entities):.2%}" if entities else "0.00%",
        "reported_uie_rows": f'{row_levels["reported_ultimate_parent"]:,}',
        "direct_parent_nearest_rows": f'{row_levels["reported_direct_parent_nearest"]:,}',
        "own_legal_economy_rows": f'{row_levels["matched_entity_legal_economy"]:,}',
        "own_hq_economy_rows": f'{row_levels["matched_entity_hq_economy"]:,}',
        "unresolved_rows": f'{row_levels["unresolved_no_gleif_entity"]:,}',
        "reported_uie_entities": f'{entity_levels["reported_ultimate_parent"]:,}',
        "nearest_economy_entities": f'{sum(count for level, count in entity_levels.items() if level not in {"reported_ultimate_parent", "unresolved_no_gleif_entity"}):,}',
        "unresolved_entities": f'{entity_levels["unresolved_no_gleif_entity"]:,}',
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive GLEIF-reported UIE or nearest available GLEIF economy for DIA/DIL entities."
    )
    parser.add_argument("--source-stem", default=DEFAULT_SOURCE_STEM)
    parser.add_argument("--stamp", default=date.today().isoformat())
    args = parser.parse_args()

    input_rows = {
        "DIL": read_csv(REPORTS / f"{args.source_stem}_dil.csv"),
        "DIA": read_csv(REPORTS / f"{args.source_stem}_dia.csv"),
    }
    assigned = {flow: [assign_economy(row) for row in rows] for flow, rows in input_rows.items()}
    entities = {flow: entity_rows(rows) for flow, rows in assigned.items()}
    summary = [summary_row(flow, assigned[flow], entities[flow]) for flow in ("DIL", "DIA")]

    stem = f"dia_dil_gleif_uie_or_nearest_economy_{args.stamp}"
    workbook = REPORTS / f"{stem}.xlsx"
    write_csv(REPORTS / f"{stem}_summary.csv", summary)
    for flow in ("DIL", "DIA"):
        write_csv(REPORTS / f"{stem}_{flow.lower()}_rows.csv", assigned[flow])
        write_csv(REPORTS / f"{stem}_{flow.lower()}_entities.csv", entities[flow])
    write_xlsx(
        workbook,
        {
            "summary": summary,
            "DIL_entities": entities["DIL"],
            "DIA_entities": entities["DIA"],
            "DIL_rows": assigned["DIL"],
            "DIA_rows": assigned["DIA"],
        },
    )
    print(summary)
    print(workbook)


if __name__ == "__main__":
    main()
