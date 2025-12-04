# Comparable Companies Engine


## Strategic Overview

### Problem
Traditional comparables selection relies on industry codes (SIC/GICS) and keyword matching, which fails when:
- Companies span multiple industries (e.g., SaaS + services)
- Business models differ despite similar products (e.g., REITs vs hospitality aggregators)
- Revenue structures are fundamentally different (e.g., subscription vs project-based)

### Solution
**Multi-signal ranking with economic signature matching:**
1. **Economic Signature** (40% weight) - Universal revenue structure matching (capacity_unit, pricing_basis, asset_intensity)
2. **Product/Customer Overlap** (25% combined) - NLP-embeddings for semantic matching
3. **Business Model Similarity** (20% combined) - Archetype/channel/delivery mode matching
4. **Soft Penalties** - Quadratic penalties for mismatches (not hard gates)

**Key Innovation:** Economic signature enables cross-industry matching while filtering anti-comps (e.g., REITs for hospitality targets).

---

## Architecture

### Pipeline Flow
```
Universe (7,900 companies)
  ↓
[Preliminary Filter] → Semantic KNN (FAISS) + Keywords → ~300 candidates
  ↓
[Shortlist] → Top 80 by preliminary score
  ↓
[Evidence Gathering] → Website + SEC 10-K + XBRL → EvidencePack
  ↓
[LLM Extraction] → Structured data (business_activity, customer_segment, economic_signature)
  ↓
[Feature Engineering] → 9 features (P, C, S, B, V, E_SIG, I, E, R)
  ↓
[Scoring] → Weighted linear + penalties + gates → Final score
  ↓
[Output] → Ranked CSV + Metadata JSONL + SHAP explanations
```

### Core Components

**1. Preliminary Filter** (`src/prelim/prelim_filter.py`)
- **Semantic KNN**: FAISS index (text-embedding-3-large) → top 300 neighbors
- **Keyword Matching**: Substring matching on business_activity/customer_segment
- **Sector/Industry**: Jaccard similarity on industry classifications
- **Combined Score**: `score_pre = w_semantic×S_fast + w_keyword×KW_score + w_sector×sector_match + w_industry×industry_match`

**2. LLM Extraction** (`src/nlp/llm_extract.py`)
- **Input**: EvidencePack (website text + SEC 10-K + XBRL)
- **Output**: Structured JSON with:
  - `business_activity`: List of products/services
  - `customer_segment`: List of customer types
  - `economic_signature`: `{capacity_unit, pricing_basis, asset_intensity_0_1, revenue_recurring_0_1, inventory_fragmentation_0_1, demand_matching_role, utilization_metric}`
  - `revenue_channels`: Dict of revenue channel percentages
  - `revenue_archetypes`: Dict of archetype percentages
- **Model**: GPT-4o with structured JSON output
- **Caching**: Persistent cache by evidence hash

**3. Feature Engineering** (`src/features/`)
- **P** (Product Overlap): NLP embeddings (cosine similarity) on business_activity, materiality-weighted
- **C** (Customer Overlap): Substring matching on customer_segment
- **S** (Segment Similarity): Cosine similarity on segment distribution vectors (portfolio-aware)
- **B** (Business Model): Services share + business_model_type similarity
- **V** (Vertical Similarity): Multi-hot encoding of vertical categories
- **E_SIG** (Economic Signature): **PRIMARY** - Cosine similarity on economic signature vectors (capacity_unit, pricing_basis, etc.)
- **I** (Industry Proximity): Jaccard similarity on industry buckets (from `industry_map.yaml`)
- **E** (Evidence Quality): Probabilistic model (10-K weighted highest: 0.9, website: 0.6, LinkedIn: 0.4)
- **R** (Recency): Linear decay (1.0 if ≤24mo, 0.0 at 60mo)

**4. Scoring System** (`src/ranker/scorer_rule.py`)

**Base Linear Score:**
```
score_linear = w_P×P + w_C×C + w_S×S + w_B×B + w_V×V + w_E_SIG×E_SIG + w_I×I + w_E×E + w_R×R + w_sic×same_sic
```

**Default Weights:**
- E_SIG: 0.40 (economic signature - PRIMARY)
- P: 0.15 (product overlap)
- S: 0.10 (segment similarity)
- B: 0.10 (business model)
- V: 0.10 (vertical)
- C: 0.10 (customer)
- I: 0.03 (industry)
- E: 0.01 (evidence)
- R: 0.01 (recency)

**Adjustments:**
1. **Interaction Penalty**: `similarity_scale = 0.3 + 0.7×max(C, segment_similarity)`, `interaction_scale = 0.3 + 0.7×(P×C)`, applied additively
2. **Model Similarity Bonus**: Archetype/channel/delivery mode similarities (configurable weights)
3. **Segment Concentration Penalty**: Entropy-based penalty for single-segment companies
4. **Discipline Penalty** (Quadratic): `penalty = (D / MIN_D)²` if `D < 0.15` (kills SaaS vs consulting mismatches)
5. **Economic Similarity Penalty** (Quadratic): `penalty = (sim / 0.30)²` if `sim < 0.30` (kills different economic engines)
6. **Business Model Penalty**: Soft penalty based on services_share mismatch

**Gates** (Hard Filters):
- `hospitality_keywords`: Filters REITs for hospitality targets
- `economic_engine`: Ensures same revenue mechanism (e.g., revenue per night vs per square foot)
- `product_hits`: Minimum product keyword matches
- `customer_hits`: Minimum customer keyword matches

**Final Score:**
```
score_100 = 100 × score_adjusted  (0-100 scale)
```

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add OPENAI_API_KEY and APIFY_API_TOKEN
```

### One-Time Setup
```bash
# Build universe (~7,900 companies)
python src/universe/build_universe.py

# Generate FAISS embeddings index (requires OpenAI API key)
python src/universe/embeddings_index.py --openai
```

### Run Pipeline
```bash
# Option 1: All-in-one (creates target.json automatically)
python comps/cli/run_pipeline.py \
  --name "ACME Corporation" \
  --url "https://www.acme.com" \
  --description "Global provider of widgets and solutions." \
  --primary-industry-classification "Technology" \
  --linkedin-url "https://www.linkedin.com/company/acme-corp/"
  --openai

# Option 2: Use existing target.json
python cli/run_pipeline.py --target data/target_awaze.json --openai
```

### Output Files
- `{target_id}_ranked.csv` - Full leaderboard with features, scores, gates
- `{target_id}_final_comps.csv` - Top 10 clean format
- `{target_id}_comps_meta.jsonl` - Per-candidate metadata with SHAP explanations
- `{target_id}_knn.csv` - KNN leaderboard (semantic similarity only, for reference)
- `{target_id}_run_summary.json` - Execution summary

---

## Configuration

### Scoring Weights (`config/scoring_config.yaml`)
```yaml
weights:
  E_SIG: 0.40  # Economic signature (PRIMARY)
  P: 0.15      # Product overlap
  C: 0.10      # Customer overlap
  S: 0.10      # Segment similarity
  B: 0.10      # Business model
  V: 0.10      # Vertical
  I: 0.03      # Industry
  E: 0.01      # Evidence
  R: 0.01      # Recency
```

### Pipeline Parameters (`config/runtime.yaml`)
```yaml
prelim_filter:
  K_semantic: 300    # Top K from FAISS
  K_keyword: 200     # Top K from keyword matching
  N_prelim: 250      # Final prelim candidates
  w_semantic: 0.5    # Weight for semantic score
  w_keyword: 0.25    # Weight for keyword score

shortlist_cap: 80    # Candidates for evidence gathering
tenk_trigger_topN: 30  # Top N get SEC 10-K
```

### Economic Signature Archetypes (`src/features/archetypes.py`)
Predefined templates:
- `hospitality_rentals_aggregator`: `{capacity_unit: "nights", pricing_basis: ["ADR", "commission"], ...}`
- `real_estate_rental_yield`: `{capacity_unit: "square_feet", pricing_basis: ["rent"], ...}`
- `consulting_services`: `{capacity_unit: "hours", pricing_basis: ["time_and_materials"], ...}`
- `industrial_oem_capital_goods`: `{capacity_unit: "units_sold", pricing_basis: ["product_sale"], ...}`

**Distance Metric**: L1 distance on numeric features + mismatch penalties on categorical features.

---

## Technical Details

### Economic Signature Matching
**Purpose**: Universal revenue structure matching that works across industries.

**Fields:**
- `capacity_unit`: `["hours", "nights", "square_feet", "units_sold", "MW", "none"]`
- `pricing_basis`: `["time_and_materials", "fixed_fee", "subscription", "ADR", "commission", "rent", "product_sale"]`
- `asset_intensity_0_1`: Asset-heavy (1.0) vs asset-light (0.0)
- `revenue_recurring_0_1`: Recurring (1.0) vs one-time (0.0)
- `inventory_fragmentation_0_1`: Fragmented (1.0) vs unified (0.0)
- `demand_matching_role`: `["none", "aggregator", "marketplace", "vertically_integrated"]`
- `utilization_metric`: `["hours_utilized", "occupancy", "throughput", "none"]`

**Similarity**: Cosine similarity on normalized vectors → 40% of final score.

### Feature Computation

**P (Product Overlap)**:
- Embed target `business_activity` and candidate `business_activity` using OpenAI embeddings
- Compute cosine similarity
- Materiality-weighted (if initiatives provided)

**C (Customer Overlap)**:
- Substring matching on `customer_segment` lists
- Normalized by max(target_segments, candidate_segments)

**S (Segment Similarity)**:
- Build segment vocabulary from all companies
- Create distribution vectors (portfolio exposure)
- Cosine similarity on vectors
- Entropy penalty for concentration

**E_SIG (Economic Signature)**:
- Extract `economic_signature` from LLM extraction (or infer from description)
- Convert to normalized vector
- Cosine similarity between target and candidate

### Penalties

**Discipline Penalty** (Quadratic):
```
if D < 0.15:
    penalty_multiplier = (D / 0.15)²
    score = score × penalty_multiplier
```
Where D = weighted similarity of (archetype, channel, delivery_mode).

**Economic Similarity Penalty** (Quadratic):
```
if overall_economic_sim < 0.30:
    penalty_multiplier = (sim / 0.30)²
    score = score × penalty_multiplier
```

**Business Model Penalty** (Linear):
```
penalty = w_pen × max(0, anchor - services_share)
score = score - penalty
```

### Gates

**Hard Gates** (all must pass):
1. `hospitality_keywords`: For hospitality targets, candidate must have keywords (hotel, resort, vacation rental, etc.)
2. `economic_engine`: Target and candidate must have same economic engine (e.g., `revenue_per_night` vs `revenue_per_square_foot`)
3. `product_hits`: Minimum product keyword matches (default: 0)
4. `customer_hits`: Minimum customer keyword matches (default: 0)

**Soft Penalties** (not gates):
- Business model mismatch → penalty
- Segment concentration → penalty
- Discipline mismatch → quadratic penalty
- Economic mismatch → quadratic penalty

---

## Output Format

### `{target_id}_ranked.csv`
Columns:
- `ticker`, `name`, `exchange`
- Features: `P`, `C`, `S`, `B`, `V`, `E_SIG`, `I`, `E`, `R`
- Scores: `score_linear`, `score_adjusted`, `score_100`
- Gates: `gate_hospitality_keywords`, `gate_economic_engine`, `passed_gates`
- Metadata: `product_hits`, `customer_hits`, `confidence_final`

### `{target_id}_comps_meta.jsonl`
Per-candidate JSON objects:
```json
{
  "ticker": "ABNB",
  "rank": 1,
  "score_100": 87.5,
  "features": {"P": 0.85, "C": 0.72, "E_SIG": 0.91, ...},
  "archetype_info": {
    "target_archetype": "hospitality_rentals_aggregator",
    "candidate_archetype": "hospitality_rentals_aggregator",
    "similarity": 0.91
  },
  "gate_details": {
    "hospitality_keywords": true,
    "economic_engine": true,
    "discipline_similarity": 0.88,
    "economic_penalty_multiplier": 1.0
  },
  "explanation": {
    "natural_language": "Ranked #1 with score 87.5. Strong economic signature match (0.91)..."
  }
}
```

---

## Caching

All data fetching is cached:
- **LLM Extractions**: Persistent cache by evidence hash
- **Embeddings**: Persistent cache (`data/cache/embedding/`)
- **SEC 10-K**: Persistent cache (`data/cache/sec/`)
- **Website**: 4-month cache
- **LinkedIn**: 4-month cache

---

## Requirements

- Python 3.8+
- OpenAI API key (for embeddings + LLM extraction)
- Apify API token (for website crawling + LinkedIn scraping)
- FAISS (for semantic search)
- XGBoost (for SHAP explanations)

---

## Key Design Decisions

1. **Economic Signature as Primary Feature (40%)**: Enables cross-industry matching while filtering anti-comps
2. **Soft Penalties vs Hard Gates**: All companies ranked, but mismatches heavily downweighted (ensures full list)
3. **Multi-Signal Approach**: Combines semantic (embeddings), structural (economic signature), and traditional (keywords) signals
4. **Explainability**: SHAP values + evidence quotes for each feature
5. **Configurable**: All weights/thresholds in config files (no hardcoding)

---

## Structure

```
comps/
├── cli/run_pipeline.py          # Main orchestrator
├── src/
│   ├── prelim/                   # Preliminary filter (FAISS KNN + keywords)
│   ├── universe/                 # Universe building, embeddings, FAISS index
│   ├── evidence/                 # Evidence gathering (website, SEC, XBRL)
│   ├── nlp/                      # LLM extraction (GPT-4o)
│   ├── features/                 # Feature computation (P, C, S, E_SIG, etc.)
│   └── ranker/                   # Scoring system (rule_score, gates, penalties)
├── config/
│   ├── scoring_config.yaml       # Feature weights, penalties, gates
│   ├── runtime.yaml              # Pipeline parameters
│   ├── industry_map.yaml         # Industry bucket mappings
│   └── model_schema.yaml         # Revenue archetypes/channels/delivery modes
└── data/
    ├── universe_us.csv           # Company universe (~7,900)
    ├── embeddings/               # FAISS index + metadata
    └── outputs/                  # Generated results
```

output metadata explanation : 
1. Basic Company Metadata
"ticker", "name", "exchange":
Directly from the universe candidates file (e.g., universe_us.csv).
Loaded at the very start of the pipeline, usually handled in the universe data loader module.
2. "confidence"
An overall confidence score for this match.
Typically calculated as a weighted combination of various feature confidences (can also be influenced by the candidate’s data completeness or specific heuristic rules).
Comes from a final aggregation function, likely in the scoring or merging module.
3. "features"
A dictionary of similarity or match scores across various dimensions:
"P": Product similarity
"C": Customer overlap
"M": Market overlap
"S": Services similarity
"I": Industry similarity (calculated via industry_map.yaml; see mapping/feature code referencing this config)
"E": Evidence score (amount and quality of supporting evidence)
"R": Recency (how recent/fresh is the data or activity)
These are computed in respective modules in src/features/, such as product_overlap.py, customer_overlap.py, industry_prox.py, etc.
4. Scores
"rule_score": Heuristic or rules-based score. Computed in the main scoring function, aggregating raw feature scores by pre-set weights and applying penalties/boosts.
"knn_score": K-nearest neighbors score (when embeddings are used).
"score_linear": Weighted linear sum of features. Calculated where the pipeline specifies weights for each feature (weighting is visible in feature_scores below).
"score_model": Score given by a machine learning model (e.g., LightGBM). If null, model either not used or couldn't calculate (see earlier explanation).
"shap": SHAP values for model explainability (filled only if SHAP calculation run; otherwise null). Generated by SHAP module when ML model is in use.
5. "explanation"
"natural_language": Human-readable summary of the match, typically generated by LLM or rules-based text synthesis (see explanation.py or LLM output handling).
"feature_scores": Detailed breakdown, for each feature:
"raw": The raw similarity/score.
"weight": Weight for this feature in the linear model.
"weighted_contribution": Product of raw and weight.
"pct_of_total": As percentage of total score.
"top_contributors": Features that contributed most to the score.
6. "contributions"
Percentage contribution of each feature to the total score (not always summing exactly to 100%—may reflect importance or normalization).
Calculated in the scoring aggregation step.
7. "evidence_snippets"
LLM-generated or extracted contetxtual quotes supporting the match.
Gathered during pipeline’s “evidence gathering” (fetch_* scripts and embeddings modules), and usually mapped in evidence_quality.py.
8. "evidence_by_feature"
Evidence split across features ("P", "C", "M", "S"), e.g., product-relevant, customer-relevant quotes.
Built during feature extraction and evidence mapping phases.
9. "business_model"
Details: Estimated by LLM or rules logic via archetype_inference.py or business_model_similarity.py.
"business_model_type": E.g., "services", "other", etc.
"services_share_estimate": Fraction of company revenue estimated to come from services.
"penalty_producty": Penalty if not “producty” enough (computed during scoring based on business model in comparison to target).
Other gates (boolean flags) are set in the business model inference or business model gating code.
10. "economic_signature", "archetype_match", etc.
Domain-specific features computed in economic modeling or archetype inference modules.
Usually filled by individual feature scripts (e.g., economic_signature.py, archetype_inference.py).
11. "concept_matches"
List of key concept hits, matched against specific keywords or themes.
Simultaneously extracted during evidence/feature extraction by a keyword matcher or LLM.
12. "segment_mix_target", "segment_mix_candidate", "segment_similarity", "dominant_segment_match"
Segment-related features, describing similarity and overlap between the target’s and candidate’s customer segments.
Computed via segment analysis modules (segment_distribution.py, etc.).
13. "evidence"
Raw LLM or web-scraped snippets cited as underlying justification for the match.
Pulled from all evidence- and extraction-related modules.
14. "confidence_final"
Aggregated overall confidence, a post-hoc numeric rollup after gates and penalties.
Produced in the last steps of scoring and exporting modules.
15. "model_meta"
Details about the model or prompt version (e.g., LLM called, or pipeline version used).
Set in the “model invocation” logic; useful for debugging reproducibility.
16. "paths", "passed_gates", "gate_business_model"
Pathways or checks the candidate passed to be considered (“gates”).
Set during the pipeline’s gating and filtering logic.
17. "timestamp"
When the scoring/output for this candidate was written.
Set during export phase.
Where Each Part Comes From in the Code
Basic fields (ticker, name, etc.):
Loaded from universe in io/export_* or initial data loader scripts.
Features & contributions:
src/features/ scripts for each aspect (product, customer, market, industry, etc.).
Business model:
src/features/archetype_inference.py, business_model_similarity.py
Scores:
Aggregated and calculated in src/ranker/scorer_rule.py, lgb_ranker.py, macro_statistical_merge.py
Evidence:
Gathered in src/evidence/ (fetch_* scripts), and mapped to features in evidence-related files.
Explanations, shap:
src/features/evidence_quality.py, src/features/model_similarity.py, potentially view_extraction.py
Meta/gates:
Final export—src/io/export_csv.py, etc.
