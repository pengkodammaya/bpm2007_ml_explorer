from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR
from src.io_helpers import load_df, save_csv, save_df
from src.uie_validation import load_validation_table, validate_uie_assignments


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate pipeline UIE assignments against ground-truth UIE labels."
    )
    parser.add_argument("ground_truth", help="Ground-truth UIE file: CSV, parquet, or Excel")
    parser.add_argument("--country", default="MY", help="Host country code for pipeline outputs, default MY")
    parser.add_argument(
        "--assignments",
        help="Optional explicit pipeline UIE assignments base path or file path",
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROCESSED_DIR / "validation"),
        help="Output directory for validation detail, summary, and confusion tables",
    )
    args = parser.parse_args()

    country = args.country.upper()
    truth = load_validation_table(args.ground_truth)

    if args.assignments:
        assignment_path = Path(args.assignments)
        if assignment_path.suffix:
            assignments = load_validation_table(assignment_path)
        else:
            assignments = load_df(assignment_path)
    else:
        assignments = load_df(PROCESSED_DIR / country.lower() / "uie_assignments")

    detail, summary, confusion = validate_uie_assignments(truth, assignments, country=country)

    out_dir = Path(args.out_dir)
    detail_base = out_dir / "uie_ground_truth_validation"
    summary_base = out_dir / "uie_ground_truth_summary"
    confusion_base = out_dir / "uie_ground_truth_confusion"

    save_df(detail, detail_base)
    save_csv(detail, detail_base.with_suffix(".csv"))
    save_df(summary, summary_base)
    save_csv(summary, summary_base.with_suffix(".csv"))
    save_df(confusion, confusion_base)
    save_csv(confusion, confusion_base.with_suffix(".csv"))

    exact_rate = float(
        summary.loc[summary["metric"].eq("exact_country_match_rate"), "value"].iloc[0]
        if not summary.empty
        else 0.0
    )
    top3_rate = float(
        summary.loc[summary["metric"].eq("top3_match_rate"), "value"].iloc[0]
        if not summary.empty
        else 0.0
    )
    print(f"[INFO] Ground-truth rows: {len(detail):,}")
    print(f"[INFO] Exact UIE-country match rate: {exact_rate:.2%}")
    print(f"[INFO] Top-3 match rate: {top3_rate:.2%}")
    print(f"[INFO] Wrote {detail_base.with_suffix('.csv')}")
    print(f"[INFO] Wrote {summary_base.with_suffix('.csv')}")
    print(f"[INFO] Wrote {confusion_base.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
