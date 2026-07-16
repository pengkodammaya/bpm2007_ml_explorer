# DIA/DIL Global GLEIF-Only Counterparty UIE Experiment

Generated: 2026-07-16
GLEIF bulk snapshot: 2026-07-15
Input: `DIA and DIL Entity Name_by Country.xlsx`

## Method

Each non-resident counterparty name is matched locally against the global GLEIF Level 1 bulk file. Unique exact normalised names are preferred. The spreadsheet immediate-country code is used only to disambiguate duplicate exact names and to constrain conservative fuzzy candidates.

A UIE is assigned only when the matched counterparty LEI has an active `IS_ULTIMATELY_CONSOLIDATED_BY` relationship in GLEIF Level 2 and the reported parent LEI has a usable country. Direct-parent links are diagnostic only. No curated group aliases, address propagation, broader graph inference, machine learning, default-immediate assignment or survey UIE is used to produce the estimate.

The survey UIE field is copied only for after-the-fact validation. UNK, ZZ and blank reference values are excluded from exact-country accuracy.

## Results

| Flow | Rows | Counterparty match coverage | GLEIF ultimate UIE coverage | Scorable assigned | Hits | Conditional hit rate |
|---|---:|---:|---:|---:|---:|---:|
| DIL | 3,567 | 26.10% | 7.96% | 284 | 196 | 69.01% |
| DIA | 2,278 | 6.94% | 1.80% | 19 | 13 | 68.42% |

## Interpretation

This is the clean GLEIF-only baseline for the spreadsheet counterparties. It measures what can be recovered from public GLEIF entity names and explicitly reported ultimate-parent relationships before any broader project methodology is added.