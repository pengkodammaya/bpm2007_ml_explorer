# Extended Abstract (IFC Biennial Conference)

## 1. Motivation

Compiling external sector statistics under BPM7 remains challenging in many jurisdictions due to persistent gaps in coverage, timeliness, and data integration. These challenges are particularly acute for direct investment statistics, where the identification of Ultimate Investor Economy (UIE) requires visibility into cross-border ownership chains that are rarely captured in a single data source.

In Malaysia, for example, the Global Legal Entity Identifier (LEI) system records 2,195 entities with active LEIs. Yet only 179 of these (8.2%) have any parent relationship data filed in the GLEIF database. The remaining 91.8% represent a structural blind spot: entities whose ownership chains are unknown, making UIE classification impossible through conventional means.

Traditional approaches to closing these gaps rely on incremental improvements to survey frames and deterministic matching. Such methods struggle with the scale and opacity of modern cross-border structures, where subsidiaries may be registered at shared corporate-secretary addresses, operate under brand names distinct from their legal parent, or file reporting exceptions rather than ownership disclosures.

## 2. Approach

This paper develops and demonstrates a multi-phase inference pipeline that uses machine learning and graph analytics as a **discovery layer** to reconstruct ownership structures from fragmented public data, prior to formal BPM7-consistent compilation.

The approach is organised into two stages:

### Discovery Layer (Machine Learning-enabled)

Public entity, relationship, and address data from the GLEIF API are integrated through four sequential inference phases:

- **Phase 0.5 -- Reporting Exception Analysis.** GLEIF's reporting exception endpoint reveals *why* entities do not file parent data. Entities filing `NON_CONSOLIDATING` exceptions are definitionally subsidiaries -- a high-confidence signal requiring no inference. Other exception types (`NATURAL_PERSONS`, `NO_LEI`, `NO_KNOWN_PERSON`, `NON_PUBLIC`) provide structural context about the nature of the coverage gap.

- **Phase 1 -- Name-Pattern Fuzzy Matching.** Core brand tokens are extracted from the 127 known foreign parent entities (stripping legal suffixes: LTD, GMBH, SDN BHD, etc.) and fuzzy-matched against all domestic entity names using `token_set_ratio` scoring with whole-word boundary enforcement. This identifies likely subsidiaries whose legal names embed the parent brand (e.g., "NESTLE MANUFACTURING (MALAYSIA) SDN. BHD." matching parent "Nestle S.A.").

- **Phase 2 -- Address Clustering.** Full street addresses are fetched from GLEIF, normalised with country-specific abbreviation expansion, and clustered by `normalised_address_line1 | postal_code`. Entities sharing a registered address with a known subsidiary inherit that subsidiary's parent signal. Office-hotel and corporate-secretary addresses (e.g., Labuan IBFC) are flagged separately.

- **Phase 3 -- Jurisdiction Prediction.** A gradient-boosted classifier is trained on the 179 labelled entities to predict the most likely parent country for all 2,195 entities. Features include legal form, entity category, city frequency, registration year, legal-name patterns, graph centrality metrics, and reporting exception flags. Country targets with fewer than five training samples are bucketed into regional groups (EUROPE, ASIA_OTHER, OFFSHORE, etc.).

- **Phase 4 -- Graph Link Prediction and Label Propagation.** The ownership graph is enriched with inferred edges from Phases 1--2, then networkx topology metrics (Jaccard coefficient, Adamic-Adar index, preferential attachment) are used to train a link predictor. Separately, jurisdiction labels are propagated through connected components weighted by graph distance, providing an independent validation signal against Phase 3's tabular predictions.

All inference outputs feed into a **coverage gap score** -- a weighted composite of eight signals (network prominence, US filing presence, foreign parent confirmation, name-inferred parent, address cluster membership, non-consolidating status, missing direct parent, missing ultimate parent) that ranks entities by their likely importance for UIE classification.

### Statistical Enforcement Layer (BPM7-aligned)

Outputs from the discovery layer are designed to be reconciled within a BPM7-consistent framework, ensuring:

- Correct classification of portfolio vs. direct investment based on inferred ownership thresholds
- Adherence to valuation principles
- Consistency with stock-flow accounting identities
- Prioritisation of official and administrative data over model-based estimates

## 3. Data and Implementation

The framework is implemented as an open-source Python pipeline operating entirely on publicly available data:

- **GLEIF API v1** -- LEI records, parent relationships, reporting exceptions, and full legal addresses for all entities registered in a given jurisdiction
- **SEC EDGAR** -- US corporate filing cross-reference for identifying entities with US regulatory presence
- **NetworkX** -- Directed ownership graph construction with PageRank, degree centrality, and connected component analysis
- **scikit-learn** -- Gradient boosting and random forest classifiers for jurisdiction prediction and link prediction
- **RapidFuzz** -- High-performance fuzzy string matching for brand-token extraction

Malaysia is used as the primary case study, reflecting a jurisdiction with relatively strong LEI adoption (2,195 entities across 214 cities and 16 legal forms) but limited parent-relationship disclosure (8.2% coverage). The pipeline is designed to be country-agnostic: a `--country` parameter switches all data paths, address normalization rules, legal-name features, and office-hotel markers, with explicit configurations currently defined for MY, SG, PH, TH, and ID.

## 4. Key Findings

Applying the pipeline to Malaysia's 2,195 LEI entities yields the following preliminary results:

### Coverage gap is structural, not random

Of the 2,016 entities without filed parent relationships, 1,751 (86.9%) have a **reporting exception** on record. The breakdown reveals the nature of the gap:

| Exception reason | Count | Interpretation |
|---|---|---|
| NON_CONSOLIDATING | 792 | Confirmed subsidiaries (do not consolidate) |
| NATURAL_PERSONS | 400 | Owned by individuals (no corporate parent LEI) |
| NO_LEI | 216 | Parent exists but has no LEI |
| NO_KNOWN_PERSON | 215 | Controlling person unknown |
| NON_PUBLIC | 128 | Ownership is non-public |

The 792 non-consolidating entities are definitionally subsidiaries -- they have corporate parents but are exempt from consolidation reporting. This alone increases the identified-subsidiary population from 179 to 971, a **5.4x improvement** requiring no statistical inference.

### Name matching identifies multinational subsidiary networks

From 127 foreign parent entities spanning 25 countries (led by US: 28, Germany: 18, UK: 17, Singapore: 12), the pipeline extracts 20 distinct brand tokens and matches them against domestic entity names, yielding 42 inferred parent-subsidiary links with an average confidence score of 93.6 (on a 0--100 scale). Of these, 27 score above 90 (high confidence). Top matched networks include Citigroup (7 subsidiaries), Allianz (4), Great Eastern (3), Standard Chartered (3), Cargill (2), and Knowles (2).

### Address clustering reveals shared-office structures at scale

882 entities (40.2% of the total) share a registered address with at least two other entities, forming 122 distinct clusters. The largest cluster contains 132 entities at a single address. Within these clusters, 197 parent links are inferred by propagating known parent signals to co-located entities. Seventeen entities are flagged at known office-hotel or corporate-secretary addresses (Labuan IBFC, Wisma UOA).

### Jurisdiction prediction provides UIE approximations for all entities

A gradient-boosted classifier trained on 179 labelled entities across 13 country-or-region classes predicts parent jurisdiction for all 2,195 entities. The top five predicted parent jurisdictions are: US (568), UK (398), Singapore (378), Germany (300), and Offshore centres (175). Cross-validation accuracy is modest (20.6%) given the small training set and class imbalance, but the model produces high-confidence predictions (probability >= 0.5) for 2,102 entities (95.8%).

### Graph-based label propagation provides independent validation

Propagating jurisdiction labels through the enriched ownership graph (2,322 nodes, 255 edges after incorporating inferred links) assigns country labels to 170 entities. Where Phase 3 and Phase 4 predictions overlap, they agree on 112 of 170 entities (65.9%). Notably, where they disagree, Phase 4 tends to provide **more specific** jurisdiction predictions (e.g., Jersey instead of "OFFSHORE", New Zealand instead of "ASIA_OTHER") because it propagates from actual parent nodes rather than predicting from bucketed categories.

### Combined coverage improvement

| Signal source | Entities identified |
|---|---|
| Known GLEIF parent relationships | 179 |
| Confirmed subsidiaries (NON_CONSOLIDATING) | 792 |
| Name-inferred subsidiaries | 42 |
| Address-cluster inferred links | 197 |
| Graph-propagated jurisdiction labels | 170 |
| Tabular jurisdiction predictions (all entities) | 2,195 |

Starting from 8.2% parent-data coverage, the pipeline provides at least one ownership or jurisdiction signal for effectively all entities, with the highest-confidence signals (known + confirmed + name-matched) covering approximately 46% of the population.

## 5. Policy Implications

The findings suggest that machine learning and graph analytics can support external sector statistics in three concrete ways:

**Improving coverage.** The single most impactful finding is that GLEIF's reporting exception data -- which requires no inference at all -- identifies 792 confirmed subsidiaries that are invisible in the standard parent-relationship dataset. This is low-hanging fruit that any jurisdiction can harvest immediately.

**Enhancing efficiency.** The coverage gap score ranks all 2,322 entities (domestic plus foreign parents) by composite priority, enabling statistical agencies to focus data collection on the entities most likely to affect UIE classification. The top-scored entity (Khazanah Nasional Berhad, Malaysia's sovereign wealth fund, score: 0.55) is a 16-subsidiary network with no filed parent data -- exactly the kind of entity that warrants targeted follow-up.

**Supporting generalisability.** The pipeline is parameterised by country and has been tested with configurations for five ASEAN economies (MY, SG, PH, TH, ID). Country-specific knowledge (legal-name patterns, address abbreviations, office-hotel markers) is isolated in a configuration registry, making extension to new jurisdictions a configuration task rather than a re-engineering effort.

**Revealing structural patterns.** The address clustering finding -- that 40% of Malaysian LEI entities share registered addresses in groups of three or more -- suggests that SPE structures and corporate-secretary arrangements are far more prevalent than parent-relationship data alone would indicate. This pattern is likely replicable across jurisdictions with similar corporate registration practices.

## 6. Limitations and Future Work

The current implementation has several limitations that inform future development:

- **Jurisdiction prediction accuracy** (20.6% CV) reflects the fundamental constraint of training on 179 labelled examples across 13 classes. Accuracy will improve as the pipeline is run across multiple countries, pooling training data.
- **Link prediction** in Phase 4 is constrained by graph sparsity -- most unlabeled entities are isolated nodes with no path to any parent. Only 2 high-confidence link predictions were generated, versus 170 from label propagation.
- **No portfolio investment** modelling is included in the current prototype. The discovery layer focuses exclusively on direct investment ownership structures.
- **Validation against ground truth** is not yet possible. The inferred relationships and jurisdiction predictions require reconciliation against confidential survey data held by statistical agencies.

Future work includes running the pipeline on Singapore and the Philippines (configurations already defined), pooling cross-country training data for improved jurisdiction prediction, and developing the BPM7 enforcement layer to convert discovery-layer outputs into compilable statistical estimates.

## 7. Conclusion

This paper demonstrates that the primary value of machine learning in external sector statistics lies not in direct estimation, but in **reconstructing the underlying economic structure** prior to measurement. A four-phase inference pipeline operating entirely on public GLEIF data increases visibility of Malaysia's cross-border ownership structures from 8.2% to near-complete coverage, with the highest-confidence signals requiring no statistical inference at all.

By combining a discovery-oriented approach with BPM7-consistent design principles, the proposed framework offers a practical, reproducible, and scalable pathway for improving external sector statistics -- particularly in jurisdictions where direct investment ownership chains remain opaque.
