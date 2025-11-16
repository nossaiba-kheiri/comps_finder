# Comparable Companies Engine

A production-ready pipeline for finding comparable public companies using semantic similarity, LLM extraction, and explainable ML ranking.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env: add OPENAI_API_KEY and APIFY_API_TOKEN

# Build universe (one-time, optional)
python src/universe/build_universe.py --limit 100  # Test first
python src/universe/build_universe.py  # Full build (~7,900 companies)

# Generate embeddings index (requires OpenAI API key)
python src/universe/embeddings_index.py --openai
```

## Pipeline Overview

```
INPUT → [Target Creation] → [Preliminary Filter] → [Evidence Gathering] → 
[LLM Extraction] → [Feature Engineering] → [Ranking & SHAP] → OUTPUT
```

## Quick Start

### Option 1: All-in-One (Recommended)

```bash
python cli/run_pipeline.py \
  --name "Company Name" \
  --url "https://company.com" \
  --description "Business description..." \
  --primary-industry-classification "Industry Name" \
  --linkedin-url "https://linkedin.com/company/company-name" \
  --openai
```

This creates `target.json` automatically and runs the full pipeline.

### Option 2: Use Existing target.json

```bash
python cli/run_pipeline.py --target data/target.json --openai
```

## Pipeline Steps

### Step 0: Target Creation (if using --name)
- **Input**: Company name, URL, business description, industry classification
- **Process**: 
  - Crawl website (Apify) → filter relevant pages
  - Scrape LinkedIn posts (last 8 months, optional)
  - Extract structured data via LLM
- **Output**: `target_{companyname}.json` (cached for reuse)

### Step 1-2: Candidate Generation & Preliminary Filter
- **Input**: `target.json`, universe (~7,900 companies)
- **Process**:
  - Semantic KNN (FAISS index, top 300)
  - Keyword matching (top 200)
  - Sector/industry filtering
  - Combined scoring
- **Output**: ~200-300 preliminary candidates

### Step 3: Shortlisting
- **Input**: Preliminary candidates
- **Process**: Select top 80 by preliminary score
- **Output**: Shortlist of 80 candidates

### Step 4: Evidence Gathering
- **Input**: Shortlist candidates
- **Process**:
  - **Website scraping** (cached 4 months)
  - **SEC 10-K** (top 30 candidates, direct SEC.gov)
  - **XBRL segment data** (optional)
- **Output**: `EvidencePack` per candidate (sources with text, URLs, types)

### Step 5: LLM Extraction
- **Input**: `EvidencePack` per candidate
- **Process**: Extract structured data via LLM:
  - `business_activity` (products/services)
  - `customer_segment` (customer types)
  - `segment_mix` (revenue breakdown)
  - `evidence` (quotes with sources)
- **Output**: Structured extraction per candidate

### Step 6: Feature Engineering
- **Input**: Target + candidate extractions
- **Process**: Compute 7 features:
  - **P**: Product overlap (materiality-weighted substring matching)
  - **C**: Customer overlap (substring matching)
  - **M**: Segment mix similarity (cosine similarity)
  - **S**: Semantic similarity (embedding cosine)
  - **I**: Industry proximity (customer industries Jaccard)
  - **E**: Evidence quality (probabilistic model, 10-K weighted highest)
  - **R**: Recency (linear decay: 1.0 if ≤24mo, 0.0 at 60mo)
- **Output**: Feature vector per candidate `(P, C, M, S, I, E, R)`

### Step 7: SHAP Explanation
- **Input**: Features `(P, C, M, S)`
- **Process**:
  - Compute linear score: `score_linear = w_P*P + w_C*C + w_M*M + w_S*S`
  - Train small XGBoost model to approximate linear score
  - Compute SHAP values: `shap_P, shap_C, shap_M, shap_S, shap_base_value`
  - Map evidence quotes to features: `evidence_by_feature` (P, C, M, S)
  - Build natural language explanation with SHAP + evidence
- **Output**: SHAP contributions + evidence quotes + explanation text

### Step 8-9: Ranking & Gates
- **Input**: Features + SHAP values
- **Process**:
  - Rule-based scoring: `rule_score = Σ(weight_i × feature_i)`
  - Apply gates (minimum thresholds)
  - Rank by score
- **Output**: Ranked candidates with scores

### Step 10: Export Results
- **Input**: Ranked candidates
- **Output**:
  - `{target_id}_ranked.csv` - All ranked candidates with features, SHAP values
  - `{target_id}_comps_meta.jsonl` - Per-candidate metadata:
    - Features: `P, C, M, S, I, E, R`
    - SHAP: `base_value, P, C, M, S`
    - Evidence: `evidence_by_feature` (quotes mapped to P, C, M, S)
    - Explanation: `natural_language` (human-readable with evidence)
    - Confidence scores, concept matches, etc.
  - `{target_id}_final_comps.csv` - Top 10 clean format (name, url, ticker, business_activity, customer_segment, sic_industry)
  - `{target_id}_run_summary.json` - Execution summary

## Input/Output System

### Input

**Required:**
- `target.json` OR basic company info (`--name`, `--url`, `--description`, `--primary-industry-classification`)

**Optional:**
- `--linkedin-url` - LinkedIn company page (for target creation)
- `--openai` - Use real OpenAI embeddings/LLM (otherwise uses mock/random)
- `--limit-candidates` - Limit for testing

**target.json Schema:**
```json
{
  "name": "Company Name",
  "url": "https://company.com",
  "business_activity": ["Product 1", "Product 2"],
  "customer_segment": ["Customer Type 1", "Customer Type 2"],
  "products": ["Specific Product Names"],
  "segment_mix": {"Segment A": 0.6, "Segment B": 0.4},
  "raw_profile_text": "Full text from website + LinkedIn",
  "primary_industry_classification": "Industry Name",
  "customer_industries": ["Vertical 1", "Vertical 2"],
  "similar_industries": ["Industry Type"],
  "mode": "all_segments"
}
```

### Output Files

**`{target_id}_ranked.csv`** - Full leaderboard:
- `ticker`, `name`, `exchange`
- Features: `P`, `C`, `M`, `S`, `I`, `E`, `R`
- SHAP: `shap_P`, `shap_C`, `shap_M`, `shap_S`, `shap_base_value`
- Scores: `score_linear`, `score_model`, `ml_score`
- Metadata: `product_hits`, `customer_hits`, `confidence_final`, `passed_gates`, `rank_ml`

**`{target_id}_comps_meta.jsonl`** - Per-candidate metadata (one JSON object per line):
- `features`: `{P, C, M, S, I, E, R}`
- `shap`: `{base_value, P, C, M, S}`
- `evidence_by_feature`: Evidence quotes mapped to P, C, M, S
- `explanation`: `{natural_language: "Ranked #X with score Y...", feature_scores: {...}}`
- `confidence`: Overall confidence score
- `evidence_snippets`: 2-3 best evidence quotes

**`{target_id}_final_comps.csv`** - Top 10 clean format:
- `name`, `url`, `exchange`, `ticker`
- `business_activity`, `customer_segment`, `sic_industry`

**`{target_id}_run_summary.json`** - Execution summary:
- Metrics: candidate counts, passed gates, execution time
- Provenance: target info, config used, pipeline version

## Example Output Explanation

From `comps_meta.jsonl`:
```json
{
  "ticker": "TTEC",
  "rank": 7,
  "score_linear": 0.74,
  "features": {"P": 0.81, "C": 0.65, "M": 0.42, "S": 0.78},
  "shap": {"P": 0.14, "C": 0.07, "M": -0.02, "S": 0.01, "base_value": 0.54},
  "evidence_by_feature": {
    "P": [{"quote": "We offer digital transformation...", "source": "website", "source_url": "..."}],
    "C": [{"quote": "Our customer base includes healthcare systems...", "source": "10K", "source_url": "..."}]
  },
  "explanation": {
    "natural_language": "Ranked #7 with score 0.74. Product similarity was a positive driver (+0.14). Evidence: 'We offer digital transformation...' Customer-segment similarity contributed (+0.07). Evidence: 'Our customer base includes...' Penalties: mix (-0.02)."
  }
}
```

## Configuration

- `config/weights.yaml` - Feature weights for scoring
- `config/runtime.yaml` - Pipeline thresholds and knobs
- `config/segments_alias.csv` - Segment name aliases
- `config/taxonomy.yaml` - Product/customer taxonomy (deprecated, using direct matching)

## Structure

```
comps/
├── cli/
│   └── run_pipeline.py          # Main orchestrator
├── src/
│   ├── prelim/                   # Preliminary filter (semantic KNN + keywords)
│   ├── universe/                 # Universe building, embeddings, candidate generation
│   ├── evidence/                 # Evidence gathering (website, SEC, LinkedIn, XBRL)
│   ├── nlp/                      # LLM extraction
│   ├── features/                 # Feature computation (P, C, M, S, I, E, R)
│   └── ranker/                   # Rule-based and ML ranking
├── config/                       # Configuration files
├── data/
│   ├── target_examples/          # Example target.json files
│   └── outputs/                  # Generated outputs (gitignored)
└── requirements.txt
```

## Requirements

- Python 3.8+
- OpenAI API key (for embeddings and LLM extraction)
- Apify API token (for website crawling and LinkedIn scraping)

See `.env.example` for API key configuration.

## Caching

All data fetching is cached:
- **Target creation**: `target_{companyname}.json` cached indefinitely
- **Website crawler**: 4-month cache
- **LinkedIn scraper**: 4-month cache
- **Embeddings**: Persistent cache (`data/cache/embedding/`)
- **SEC 10-K**: Persistent cache (`data/cache/sec/`)

Use `--force` flag to recreate target.json even if cached.

## Failure Handling

The pipeline gracefully handles failures at each stage:
- Missing evidence → lower E score, continues with available data
- LLM extraction fails → mock extraction, lower confidence
- API rate limits → cached data used, warnings logged
- Missing features → neutral defaults (0.0 for P/C, 0.5 for M/S/I)

Check `{target_id}_run_summary.json` for execution status and warnings.
