# Directional UIE Result Set

This project reports UIE results in two directions.

## IN: Inward UIE

IN answers: for entities located in a host economy, which ultimate investor
economies are assigned by the methodology?

For Malaysia:

```text
host_country = MY
destination_country = MY
source_country = uie_country
```

Foreign inward UIE rows exclude domestic UIE and unknown/unassigned UIE:

```text
uie_country is known
uie_country != host_country
```

The inward density measure is:

```text
foreign-UIE LEI count in host / inward FDI stock in USD billions
```

## OUT: Outward UIE

OUT answers: for an ASEAN source economy, where do its ultimate-investor
entities appear abroad in the crawled host economies?

For Malaysia:

```text
source_country = MY
destination_country = host_country
host_country != MY
uie_country = MY
```

The outward result set is currently scoped to ASEAN-observed hosts. It should
not be read as global outward coverage unless the host crawl universe is also
global.

The outward density measure is:

```text
source-economy-owned LEI count observed in destination / outward FDI stock to
that destination in USD billions
```

## Evidence Buckets

The directional tables retain the original `evidence_tier` and add a compact
`evidence_bucket`:

```text
A_hard_or_verified
B_parent_or_exception
C_name_inferred
D_address_inferred
E_model_only
U_unassigned
Z_other
```

This allows headline results to be shown either as all methodology outputs or
as evidence-backed outputs excluding model-only rows.

## Generated Files

Run:

```powershell
py -3.13 scripts\generate_directional_uie_results.py
```

Outputs are written to `reports/directional_uie/` as timestamped CSV and
Parquet files:

```text
asean_inward_assignments
asean_inward_foreign_assignments
asean_inward_summary
asean_inward_density
asean_outward_assignments
asean_outward_summary
asean_outward_bilateral_density
asean_outward_source_density
malaysia_inward_assignments
malaysia_inward_foreign_assignments
malaysia_inward_summary
malaysia_outward_assignments
malaysia_outward_summary
malaysia_outward_bilateral_density
```

The same country-specific pattern is also generated for each ASEAN economy:

```text
{country}_inward_assignments
{country}_inward_foreign_assignments
{country}_inward_summary
{country}_inward_density
{country}_outward_assignments
{country}_outward_summary
{country}_outward_bilateral_density
```

For example, `sg_inward_summary` applies the Malaysia inward logic to Singapore
as host, while `sg_outward_summary` reports Singapore-owned LEIs observed in
other crawled ASEAN hosts.
