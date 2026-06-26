# ASEAN GLEIF/UIE Snapshot

Snapshot date: 2026-06-26  
Algorithm: current BPM2007 UIE assignment pipeline using GLEIF relationships, reporting exceptions, address/name inference, CTOS-assisted Malaysia signals where available, and model-based fallback scoring.  
Coverage: Indonesia, Malaysia, the Philippines, Singapore, Thailand, Brunei, Vietnam, Laos, Myanmar, Cambodia, and Timor-Leste.

This snapshot is provisional. Singapore reporting exceptions were still refreshing when this file was prepared, so Singapore-related evidence quality may improve after the next rebuild. Timor-Leste is included in the regional coverage set but has no inward or outward rows in the current directional result pack.

## Headline Counts

| Measure | Count |
|---|---:|
| Foreign inward UIE rows across covered ASEAN hosts | 16,673 |
| Hard/verified inward rows | 2,287 |
| Inferred inward rows | 14,386 |
| Model-only inward rows | 10,703 |
| Hard/verified inward share | 13.7% |
| Model-only inward share | 64.2% |
| ASEAN-observed outward rows | 714 |
| Hard/verified outward rows | 150 |
| Inferred outward rows | 564 |
| Model-only outward rows | 381 |
| Hard/verified outward share | 21.0% |
| Model-only outward share | 53.4% |

## Inward UIE Density by Host

Inward density is defined as:

`foreign-owned LEI count in host / inward FDI stock in USD billions`

| Host | Foreign UIE rows | Inward LEI density | Hard/verified share | Model-only share |
|---|---:|---:|---:|---:|
| PH | 694 | 8.66 | 22.9% | 76.4% |
| MY | 1,877 | 6.68 | 16.0% | 77.1% |
| TH | 2,281 | 5.56 | 11.9% | 86.9% |
| BN | 42 | 5.42 | 16.7% | 76.2% |
| ID | 1,402 | 3.51 | 18.5% | 80.2% |
| SG | 9,969 | 3.07 | 11.6% | 53.3% |
| VN | 345 | 2.83 | 34.8% | 64.1% |
| LA | 21 | 1.29 | 4.8% | 95.2% |
| KH | 36 | 0.56 | 25.0% | 75.0% |
| MM | 6 | 0.53 | 16.7% | 83.3% |
| TL | 0 | n/a | n/a | n/a |

## ASEAN-Observed Outward UIE

Outward density is defined as:

`count of entities abroad assigned to the source country as UIE / outward FDI stock in USD billions`

This is an ASEAN-observed measure only. It does not claim to capture all global outward investment by the source country.

| Source | ASEAN destinations observed | Outward observed rows | Outward observed density | Hard/verified share | Model-only share |
|---|---:|---:|---:|---:|---:|
| SG | 5 | 412 | 2.11 | 15.5% | 81.3% |
| MY | 6 | 108 | 0.74 | 25.9% | 40.7% |
| PH | 1 | 66 | 8.31 | 16.7% | 0.0% |
| ID | 1 | 66 | 0.76 | 16.7% | 0.0% |
| TH | 4 | 56 | 0.74 | 58.9% | 3.6% |
| VN | 1 | 6 | 2.20 | 50.0% | 0.0% |
| BN | 0 | 0 | n/a | n/a | n/a |
| KH | 0 | 0 | n/a | n/a | n/a |
| LA | 0 | 0 | n/a | n/a | n/a |
| MM | 0 | 0 | n/a | n/a | n/a |
| TL | 0 | 0 | n/a | n/a | n/a |

## Top Assigned Ultimate Investor Economies

These are top UIE economies across all covered ASEAN hosts, aggregated from the inward summary.

| UIE economy | LEI rows | Hard/verified | Inferred | Model-only |
|---|---:|---:|---:|---:|
| US | 3,017 | 498 | 2,519 | 2,048 |
| EUROPE | 2,025 | 0 | 2,025 | 2,025 |
| DE | 1,976 | 258 | 1,718 | 1,658 |
| GB | 1,864 | 235 | 1,629 | 1,286 |
| KY | 1,211 | 91 | 1,120 | 593 |
| IN | 1,022 | 134 | 888 | 511 |
| OTHER | 581 | 0 | 581 | 581 |
| AU | 433 | 31 | 402 | 221 |
| SG | 412 | 64 | 348 | 335 |
| FR | 383 | 109 | 274 | 69 |

## Interpretation

The regional snapshot supports the project goal: identifying ultimate investor economy signals from GLEIF-observed entities and making those signals comparable across ASEAN hosts. Singapore dominates absolute inward LEI volume, but density normalisation changes the reading: the Philippines and Malaysia show higher foreign-UIE LEI density relative to inward FDI stock than Singapore.

Malaysia is currently one of the strongest operational use cases. It has 1,877 foreign inward UIE rows and an inward density of 6.68 LEIs per USD billion of inward FDI. The largest assigned UIE economies for Malaysia are the United States, Singapore, Germany, and the United Kingdom. This directly supports the intended question of whether an investment routed through Singapore may have a non-Singapore ultimate investor economy.

Evidence quality remains the main caveat. Across the region, model-only assignments account for 64.2% of inward rows. The dataset is therefore best presented as a tiered-confidence experimental snapshot rather than a definitive register of ultimate ownership.

## Related Files

- `reports/directional_uie/asean_inward_summary_2026-06-26.csv`
- `reports/directional_uie/asean_inward_density_2026-06-26.csv`
- `reports/directional_uie/asean_outward_summary_2026-06-26.csv`
- `reports/directional_uie/asean_outward_source_density_2026-06-26.csv`
- `reports/directional_uie/asean_country_flow_graph_2026-06-26.html`
- `reports/directional_uie/malaysia_in_out_graph_2026-06-26.html`
