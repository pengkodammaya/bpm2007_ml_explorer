# Malaysia Inward ASEAN Direct-Investor to UIE Bridge

Snapshot date: 2026-06-30  
Host economy: Malaysia  
Purpose: identify cases where the direct ASEAN parent of a Malaysia-hosted entity differs from the assigned ultimate investor economy (UIE).

## Definition

This bridge uses the current UIE assignment output and keeps Malaysia-hosted entities where:

- `entity_country = MY`
- `direct_parent_country` is an ASEAN country
- `direct_parent_country != MY`
- `uie_country` is assigned

This is a hard/direct-parent bridge. It does not infer a direct investor country where GLEIF does not report or infer a direct parent. As a result, the current output is narrower than the full Malaysia inward UIE file.

## Result

The current data identifies **23 Malaysia-hosted entities** with an ASEAN direct parent outside Malaysia. All 23 have **Singapore (`SG`)** as the direct parent country.

| Direct parent country | Rows | UIE differs from direct parent | Non-ASEAN UIE rows | Pass-through share |
|---|---:|---:|---:|---:|
| SG | 23 | 11 | 11 | 47.8% |

## Singapore Direct Parent Breakdown

| Direct parent country | Assigned UIE | Evidence bucket | Rows |
|---|---|---|---:|
| SG | SG | A_hard_or_verified | 11 |
| SG | US | A_hard_or_verified | 3 |
| SG | JP | A_hard_or_verified | 2 |
| SG | GB | A_hard_or_verified | 2 |
| SG | DE | A_hard_or_verified | 2 |
| SG | SG | B_parent_or_exception | 1 |
| SG | NL | A_hard_or_verified | 1 |
| SG | IT | B_parent_or_exception | 1 |

## Interpretation

The result confirms the pass-through concept. In the current Malaysia data, Singapore can appear as the direct parent country while the assigned UIE is elsewhere. Of the 23 Malaysia entities with an observed Singapore direct parent, 11 are assigned to a non-Singapore UIE: Germany, the United States, Japan, the United Kingdom, the Netherlands, or Italy.

This directly supports the operational use case:

`foreign UIE -> Singapore entity -> Malaysia entity`

For example, a Malaysia entity can have a Singapore direct parent while the UIE is assigned to the United States, Germany, Japan, or the United Kingdom.

## Caveat

This table is not a top-five ASEAN direct-investor ranking because the current hard direct-parent layer only observes Singapore as an ASEAN direct parent into Malaysia. Other ASEAN countries may still be relevant in actual FDI data, but they are not observed as direct GLEIF parent countries in this Malaysia assignment set.

For broader Malaysia inward UIE analysis, use `malaysia_inward_foreign_assignments_2026-06-30.csv`. For direct-investor pass-through analysis, use the bridge files listed below.

## Related Files

- `reports/directional_uie/malaysia_inward_asean_direct_investor_uie_bridge_2026-06-30.csv`
- `reports/directional_uie/malaysia_inward_asean_direct_investor_uie_summary_2026-06-30.csv`
- `reports/directional_uie/malaysia_inward_asean_direct_investors_2026-06-30.csv`
- `reports/directional_uie/malaysia_inward_top5_asean_direct_investor_uie_bridge_2026-06-30.csv`
- `reports/directional_uie/malaysia_inward_top5_asean_direct_investor_uie_summary_2026-06-30.csv`
