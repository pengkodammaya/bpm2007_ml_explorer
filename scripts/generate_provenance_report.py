from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
TODAY = datetime.now().date().isoformat()
COUNTRIES = ["MY", "SG", "TH", "ID", "PH", "VN", "KH", "BN", "LA", "MM", "TL"]


def row_count(path: Path) -> int | str:
    if not path.exists():
        return "missing"
    try:
        if path.suffix == ".parquet":
            return len(pd.read_parquet(path))
        if path.suffix == ".csv":
            return sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1
        if path.suffix == ".xlsx":
            return "workbook"
    except Exception as exc:
        return f"unreadable: {exc.__class__.__name__}"
    return ""


def stamp(path: Path) -> str:
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def add(rows: list[dict], path: Path, role: str, upstream: str, used_in: str, notes: str = "") -> None:
    rows.append(
        {
            "file": rel(path),
            "exists": path.exists(),
            "rows": row_count(path),
            "last_modified": stamp(path),
            "role": role,
            "upstream_source": upstream,
            "used_in": used_in,
            "notes": notes,
        }
    )


def manifest_rows() -> list[dict]:
    rows: list[dict] = []
    for country in COUNTRIES:
        code = country.lower()
        add(
            rows,
            DATA / "raw" / f"gleif_{code}_lei.parquet",
            f"{country} domestic LEI universe",
            "GLEIF API LEI records endpoint",
            "raw coverage table; UIE pipeline; deck country snapshot",
            "Country-linked legal-entity records. MY raw file was refreshed before final relationship refresh completed.",
        )
        add(
            rows,
            DATA / "interim" / f"gleif_{code}_relationships.parquet",
            f"{country} completed parent relationships",
            "GLEIF API relationship endpoints",
            "UIE evidence ladder; graph/scoring; relationship counts",
            "Completed relationship output. If WIP/scanned files exist, a refresh is still incomplete.",
        )
        add(
            rows,
            DATA / "interim" / f"gleif_{code}_relationships_wip.parquet",
            f"{country} relationship refresh checkpoint",
            "GLEIF API relationship endpoints",
            "provenance caveat only",
            "Checkpoint artifact, not a final input unless relationship scan completes.",
        )
        add(
            rows,
            DATA / "interim" / f"gleif_{code}_relationships_scanned.parquet",
            f"{country} scanned-LEI checkpoint",
            "GLEIF API relationship endpoints",
            "provenance caveat only",
            "Tracks scanned LEIs during resumable relationship refresh.",
        )
        add(
            rows,
            DATA / "raw" / f"gleif_{code}_related_lei.parquet",
            f"{country} related parent LEI records",
            "GLEIF API LEI record lookup by LEI",
            "parent country/name enrichment; UIE assignment",
        )
        add(
            rows,
            DATA / "interim" / f"gleif_{code}_reporting_exceptions.parquet",
            f"{country} reporting exceptions",
            "GLEIF API reporting-exception endpoints",
            "exception counts; UIE evidence/context",
        )
        add(
            rows,
            DATA / "processed" / code / "uie_assignments.parquet",
            f"{country} derived UIE assignments",
            "Pipeline-derived from GLEIF records, relationships, exceptions, inference outputs and manual overrides",
            "Word/deck UIE counts; density numerator",
        )
        add(
            rows,
            DATA / "processed" / code / "investor_economy_summary.parquet",
            f"{country} investor economy summary",
            "Pipeline-derived UIE aggregation",
            "supporting country summaries",
        )

    add(
        rows,
        DATA / "processed" / "gleif_bilateral_entity_counts.csv",
        "Bilateral host-parent LEI entity counts",
        "Pipeline-derived from UIE/relationship country assignments",
        "CDIS validation and Spearman correlations",
    )
    add(
        rows,
        DATA / "dataset_2026-04-22T22_38_34.423483046Z_DEFAULT_INTEGRATION_IMF.STA_DIP_12.0.1.csv",
        "IMF Direct Investment Positions by Counterpart Economy extract",
        "IMF Data: DIP, formerly CDIS",
        "inward FDI denominator; density; CDIS validation",
        "Local extract used for 2023 all-financial-instrument, all-entity net positions.",
    )
    add(
        rows,
        DATA / "processed" / "cdis_validation_output.txt",
        "CDIS validation text output",
        "Local validation script output",
        "cross-check against deck/writeup validation claims",
    )
    add(
        rows,
        DATA / "processed" / "cdis_validation_tables.xlsx",
        "CDIS validation workbook",
        "Local validation script output",
        "supporting validation tables",
    )
    add(
        rows,
        DATA / "processed" / "my" / "uie_review_targets.parquet",
        "Malaysia UIE review queue",
        "Pipeline-derived priority queue",
        "main review queue slide and writeup section",
    )
    add(
        rows,
        DATA / "processed" / "my" / "uie_product_vehicle_targets.parquet",
        "Malaysia product-vehicle audit queue",
        "Pipeline-derived entity-type rule output",
        "fund/product vehicle methodology and audit trail",
    )
    add(
        rows,
        DATA / "processed" / "my" / "entity_match_benchmark.parquet",
        "Malaysia GLEIF search benchmark",
        "GLEIF full-text search API via local benchmark function",
        "GLEIF search benchmark slide and writeup section",
    )
    add(
        rows,
        DATA / "manual" / "uie_overrides.csv",
        "Manual UIE overrides",
        "Analyst-entered source-backed overrides",
        "UIE evidence ladder; known/source-backed cases",
    )
    add(
        rows,
        DATA / "manual" / "uie_review_status.csv",
        "Manual review statuses",
        "Analyst-entered review classifications",
        "review queue classification",
    )
    add(
        rows,
        ROOT / "scripts" / "generate_conference_outputs.py",
        "Document/deck generation script",
        "Local code",
        "Word and PowerPoint generation; derived CSV exports",
    )
    add(
        rows,
        REPORTS / f"asean_uie_bis_style_writeup_outward_density_provenance_fund_rule_{TODAY}.docx",
        "Current BIS-style writeup with outward density and provenance",
        "Generated artifact",
        "primary Word output",
    )
    add(
        rows,
        REPORTS / f"asean_uie_bis_style_deck_sceptical_readers_provenance_fund_rule_{TODAY}.pptx",
        "Current skeptical-reader deck with provenance slides",
        "Generated artifact",
        "primary PowerPoint output",
    )
    add(
        rows,
        REPORTS / f"conference_outward_bilateral_{TODAY}.csv",
        "Outward bilateral density audit table",
        "Derived from GLEIF bilateral counts joined to IMF DIP/CDIS outward positions",
        "outward-density writeup and deck slides",
    )
    add(
        rows,
        REPORTS / f"conference_outward_source_summary_{TODAY}.csv",
        "Outward source summary audit table",
        "Derived from matched intra-ASEAN outward bilateral pairs",
        "ASEAN outward-density deck summary",
    )
    add(
        rows,
        ROOT / "src" / "pipeline.py",
        "Core pipeline implementation",
        "Local code",
        "UIE assignment, review queue, crawl orchestration",
    )
    add(
        rows,
        ROOT / "src" / "gleif.py",
        "GLEIF API client and normalisation",
        "Local code",
        "GLEIF pulls and search benchmark",
    )
    return rows


def doc_p(text: str = "", style: str | None = None) -> str:
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{style_xml}<w:r><w:t>{escape(str(text))}</w:t></w:r></w:p>"


def doc_table(rows: list[list[str]]) -> str:
    xml = ["<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"0\" w:type=\"auto\"/></w:tblPr>"]
    for row in rows:
        xml.append("<w:tr>")
        for cell in row:
            xml.append(f"<w:tc><w:tcPr><w:tcW w:w=\"1800\" w:type=\"dxa\"/></w:tcPr>{doc_p(cell)}</w:tc>")
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


DOCX_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

DOCX_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""


def build_doc(manifest: pd.DataFrame) -> Path:
    REPORTS.mkdir(exist_ok=True)
    rows = [["File", "Rows", "Modified", "Role", "Used in"]]
    for _, row in manifest.iterrows():
        if row["exists"] is True or row["exists"] == "True":
            rows.append(
                [
                    row["file"],
                    str(row["rows"]),
                    row["last_modified"],
                    row["role"],
                    row["used_in"],
                ]
            )

    body = []
    body.append(doc_p("Data Provenance for ASEAN UIE Word Document and Deck", "Title"))
    body.append(doc_p(f"Generated {TODAY}. Covers the current BIS-style writeup and deck artifacts."))
    body.append(doc_p("External Source Provenance", "Heading1"))
    body.append(
        doc_p(
            "GLEIF data are pulled through the GLEIF API documentation entry point at https://api.gleif.org/docs. "
            "The local pipeline uses LEI records, relationship endpoints, reporting-exception endpoints and full-text search results."
        )
    )
    body.append(
        doc_p(
            "IMF macro-validation data come from the IMF Data page for Direct Investment Positions by Counterpart Economy "
            "(DIP, formerly CDIS): https://data.imf.org/en/datasets/IMF.STA:DIP. The IMF page describes DIP/CDIS as covering "
            "inward positions by immediate investor economy and outward positions by immediate investment economy."
        )
    )
    body.append(doc_p("Generated Artifacts Covered", "Heading1"))
    body.append(
        doc_p(
            "Primary generated artifacts: reports/asean_uie_bis_style_writeup_2026-06-22.docx and "
            "reports/asean_uie_bis_style_deck_2026-06-22.pptx. Supporting CSV exports in reports/conference_*.csv are derived "
            "from the same local inputs and are included to make the figures easier to audit."
        )
    )
    body.append(doc_p("Important Caveats", "Heading1"))
    body.append(
        doc_p(
            "The report and deck use the current processed outputs available at generation time. Background GLEIF crawls may still "
            "be updating raw or checkpoint files. In particular, WIP/scanned relationship checkpoint files indicate that a country "
            "refresh is not yet fully finalized; final UIE tables should be regenerated after crawls complete."
        )
    )
    body.append(
        doc_p(
            "CTOS data, EDGAR enrichment and proposed GEM evidence are not treated as core evidence for the current deck/writeup. "
            "CTOS may appear in some derived UIE tables as a Malaysia incorporation signal, but it is not used as proof of Malaysian ownership."
        )
    )
    body.append(doc_p("Local File Manifest", "Heading1"))
    body.append(doc_table(rows))

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        + "".join(body)
        + '<w:sectPr><w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720"/></w:sectPr>'
        "</w:body></w:document>"
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:b/><w:sz w:val="34"/></w:rPr></w:style>'
        '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:pPr><w:spacing w:before="220" w:after="100"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>'
        "</w:styles>"
    )
    path = REPORTS / f"asean_uie_data_provenance_embedded_fund_rule_{TODAY}.docx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", DOCX_CONTENT_TYPES)
        z.writestr("_rels/.rels", DOCX_RELS)
        z.writestr("word/document.xml", document)
        z.writestr("word/styles.xml", styles)
    return path


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    manifest = pd.DataFrame(manifest_rows())
    csv_path = REPORTS / f"asean_uie_data_provenance_manifest_embedded_fund_rule_{TODAY}.csv"
    manifest.to_csv(csv_path, index=False)
    docx_path = build_doc(manifest)
    print(csv_path)
    print(docx_path)


if __name__ == "__main__":
    main()
