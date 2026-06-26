# Project Plan

## Current Focus

Improve Ultimate Investor Economy (UIE) reliability by reducing model-only
assignments and adding source-backed evidence layers.

## Next Work Items

### 1. GLEIF Search Benchmark

Benchmark GLEIF's own LEI search/matching behaviour against the top UIE review
targets.

Output:

- `data/processed/my/entity_match_benchmark.csv`

Planned fields:

- `lei`
- `legal_name`
- `current_uie_country`
- `current_uie_source`
- `gleif_candidate_lei`
- `gleif_candidate_name`
- `gleif_candidate_country`
- `gleif_match_rank`
- `gleif_match_method`
- `review_flag`

Purpose:

- Test whether GLEIF search retrieves better candidate legal entities.
- Use GLEIF as candidate generation, not automatic UIE assignment.
- Flag cases where current Phase 3/model UIE conflicts with plausible GLEIF
  candidates.

### 2. Global Energy Monitor Evidence Layer

Add Global Energy Monitor (GEM) as a sector-specific ownership evidence source
for energy and heavy-industry entities.

Output:

- `data/processed/my/gem_energy_matches.csv`

Planned fields:

- `lei`
- `legal_name`
- `gem_entity_id`
- `asset_name`
- `asset_country`
- `asset_type`
- `owner_name`
- `owner_country`
- `ownership_percent`
- `ultimate_parent_name`
- `ultimate_parent_country`
- `gem_source_url`
- `match_method`

Evidence handling:

- Initial tier: `gem_asset_ownership_candidate`
- After analyst review: `manual_verified`

Purpose:

- Strengthen UIE calls for PETRONAS, TNB, YTL Power, Malakoff, Sarawak Energy,
  LNG/gas/power assets, and other energy-sector entities.
- Add asset materiality context such as asset count, capacity, or project status.
- Use GEM ownership chains as supporting evidence, while reconciling against
  BPM7 UIE rules.

### 3. GEM-to-LEI Mapping

When the GEM-GLEIF mapping file is available, ingest the mapping between GEM
Entity ID and LEI.

Purpose:

- Avoid fragile name-only matching where certified GEM-to-LEI mappings exist.
- Connect asset ownership evidence directly to LEI entities in the UIE pipeline.

### 4. Fund/Product Vehicle Rule

Develop a separate review rule for fund and product vehicles.

Current queue signal:

- `363 / 500` top Malaysia review targets are classified as
  `fund_or_product_vehicle`.

Purpose:

- Avoid treating fund products like operating companies.
- Decide whether to assign UIE based on manager, sponsor, fund domicile, or
  another BPM7-consistent rule.

### 5. Source-Backed Manual Overrides

Continue adding verified overrides for high-priority non-fund entities.

Current next targets:

- KAF Investment Bank Berhad
- Lion Industries Corporation Berhad
- ABRDN ISLAMIC MALAYSIA SDN. BHD.
- ACB RESOURCES BERHAD
- AET DP SHUTTLE TANKERS SDN. BHD.
- AHAM Asset Management Berhad
- TELEKOM MALAYSIA BERHAD
- Axiata Group Berhad

Tracking metrics:

- Increase `known_uie_share`
- Reduce `phase3_fallback_share`
- Reduce high-priority `todo` rows in `uie_review_targets.csv`

### 6. Malaysia Top-Investor Chain Bridge

Build an inward investor-chain bridge for Malaysia's top 5 immediate investor
economies, using the current UIE methodology.

Scope:

- Host economy: Malaysia only.
- Immediate-investor set: top 5 foreign direct-parent economies observed in
  Malaysia LEI relationship data.
- Current candidates from refreshed MY data: `US`, `DE`, `GB`, `SG`, `HK`.

Output:

- `data/processed/my/top_inward_investor_chain_bridge.csv`

Planned fields:

- `lei`
- `legal_name`
- `immediate_parent_lei`
- `immediate_parent_name`
- `immediate_parent_country`
- `next_parent_lei`
- `next_parent_name`
- `next_parent_country`
- `ultimate_parent_lei`
- `ultimate_parent_name`
- `ultimate_parent_country`
- `final_uie_country`
- `uie_source`
- `chain_resolution_status`

Resolution statuses:

- `resolved_to_known_ultimate`
- `resolved_to_upstream_ultimate`
- `resolved_to_upstream_direct`
- `immediate_parent_is_final_uie`
- `no_upstream_found`
- `manual_review_needed`

Purpose:

- Separate immediate investor economy from ultimate investor economy.
- Identify Singapore, Hong Kong, and other conduit/intermediate-parent cases.
- Quantify how much of Malaysia's observed inward LEI ownership remains with
  the immediate parent economy versus resolves to another ultimate economy.

### 7. Malaysia Outward Top-Destination Chain Bridge

Build an outward bridge for the top 5 economies where Malaysia-owned LEI
entities appear abroad.

Scope:

- Source economy: Malaysia.
- Destination set: top 5 foreign host economies where current LEI/UIE outputs
  identify Malaysian ownership or Malaysian upstream parent evidence.

Output:

- `data/processed/my/top_outward_destination_chain_bridge.csv`

Planned fields:

- `destination_country`
- `foreign_entity_lei`
- `foreign_entity_name`
- `immediate_parent_lei`
- `immediate_parent_name`
- `immediate_parent_country`
- `malaysia_parent_lei`
- `malaysia_parent_name`
- `malaysia_parent_role`
- `ultimate_parent_lei`
- `ultimate_parent_name`
- `ultimate_parent_country`
- `chain_resolution_status`

Purpose:

- Show where Malaysian groups appear as owners or upstream parents abroad.
- Distinguish direct Malaysian ownership from chains where Malaysia is an
  intermediate parent.
- Produce an outward counterpart to the inward Malaysia investor bridge.

### 8. Ground-Truth UIE Validation and Supervised Enhancement

Use user-supplied ground-truth UIE data as a controlled benchmark and, only in
a separate mode, as supervised training data.

Primary use:

- Treat ground truth as an external validation set for the public-data UIE
  discovery pipeline.
- Measure exact UIE-country accuracy, top-3 model accuracy, confusion by
  country, and error rates by evidence tier.
- Identify high-confidence wrong assignments and systematic over-prediction of
  broad buckets such as `EUROPE`, `OTHER`, `OFFSHORE`, and `ASIA_OTHER`.

Planned validation output:

- `data/processed/validation/uie_ground_truth_validation.csv`
- `data/processed/validation/uie_ground_truth_summary.csv`

Planned fields:

- `lei`
- `legal_name`
- `ground_truth_uie_country`
- `ground_truth_uie_name`
- `pipeline_uie_country`
- `pipeline_uie_source`
- `pipeline_evidence_tier`
- `pipeline_confidence`
- `is_exact_country_match`
- `is_top3_match`
- `error_type`
- `review_recommendation`

Leakage guardrails:

- Keep public-data-only UIE results separate from supervised-enhanced results.
- Do not use the same ground-truth rows for both model training/tuning and final
  evaluation.
- Prefer grouped splits by corporate family or parent group, not purely random
  row splits, to avoid sister-company leakage.
- Preserve a locked holdout set that is never used for feature selection,
  threshold tuning, prompt design, or model selection.

Optional supervised mode:

- Add ground-truth UIE labels to the Phase 3 jurisdiction model only after a
  train/validation/test split is defined.
- Report supervised-enhanced outputs separately from public-data discovery
  outputs.
- Use the validation set to calibrate predicted probabilities and decide when
  model evidence should be downgraded to review-only.

Publication framing:

- Ground truth is first an audit and calibration layer for the public-data
  discovery method.
- Any model trained on ground truth is a separate supervised extension, not the
  baseline public-data methodology.
