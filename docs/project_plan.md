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

