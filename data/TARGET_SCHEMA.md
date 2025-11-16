# Target JSON Schema Guide

## Overview
The `target.json` file defines the company you want to find comparable companies for. It contains information about the target company's products, customers, segments, and profile.

## Schema

```json
{
  "name": "string (required)",
  "url": "string (optional)",
  "products": ["string", ...],  // required
  "customers": ["string", ...],  // required
  "segment_mix": {"segment": weight, ...},  // optional
  "text_profile": "string (required)",
  "mode": "string (optional, default: 'all_segments')",
  "industry": "string (optional)"
}
```

## Field Descriptions

### Required Fields

1. **`name`** (string)
   - Company name or identifier
   - Example: `"Stripe"` or `"Target Company"`
   - Used for output file naming and logging

2. **`products`** (array of strings)
   - List of products/services offered
   - Should match taxonomy.yaml product aliases for best results
   - Example: `["Payment Processing", "Transaction APIs"]`
   - Used for:
     - Product overlap scoring (P)
     - Cross-sector industry mapping
     - Keyword matching

3. **`customers`** (array of strings)
   - List of customer types/verticals
   - Should match taxonomy.yaml customer aliases for best results
   - Example: `["Banks", "Retailers"]`
   - Used for:
     - Customer overlap scoring (C)
     - Keyword matching

4. **`text_profile`** (string)
   - 2-3 sentence description combining products and customers
   - Should be comprehensive and descriptive
   - Example: `"Stripe is a fintech company offering payment processing APIs to banks, retailers, and online merchants. We provide transaction services and payment infrastructure for e-commerce platforms."`
   - Used for:
     - Semantic similarity scoring (S)
     - Embedding-based candidate generation (KNN)

### Optional Fields

5. **`url`** (string)
   - Company website URL
   - Example: `"https://stripe.com"`
   - Used for context (not actively fetched)

6. **`segment_mix`** (object)
   - Revenue breakdown by segment
   - Keys: segment names (Healthcare, Education, Commercial, Public, Financial Services, Retail)
   - Values: weights (0.0-1.0, should sum to 1.0)
   - Example: `{"Healthcare": 0.5, "Education": 0.3, "Commercial": 0.2}`
   - Used for:
     - Segment mix similarity scoring (M)
     - Dominant segment matching

7. **`mode`** (string)
   - Scoring mode (must match a mode in weights.yaml)
   - Default: `"all_segments"`
   - Example: `"all_segments"`
   - Used for:
     - Selecting feature weights
     - Applying gates

8. **`industry`** (string)
   - Industry classification (optional)
   - Example: `"Financial Services"` or `"Technology"`
   - Used for:
     - Industry proximity scoring (I)
     - If not provided, defaults to 0.5

## Examples

### Example 1: Payment Processing Company
```json
{
  "name": "Stripe",
  "url": "https://stripe.com",
  "products": ["Payment Processing", "Transaction APIs", "Payment Solutions"],
  "customers": ["Banks", "Retailers", "E-commerce"],
  "segment_mix": {
    "Financial Services": 0.4,
    "Retail": 0.3,
    "Commercial": 0.3
  },
  "text_profile": "Stripe is a fintech company offering payment processing APIs to banks, retailers, and online merchants. We provide transaction services and payment infrastructure for e-commerce platforms and financial institutions.",
  "mode": "all_segments"
}
```

### Example 2: Cloud Computing Company
```json
{
  "name": "CloudCorp",
  "url": "https://cloudcorp.com",
  "products": ["Cloud Services", "SaaS", "Cloud Infrastructure"],
  "customers": ["Enterprises", "Businesses", "Organizations"],
  "segment_mix": {
    "Commercial": 0.6,
    "Education": 0.2,
    "Healthcare": 0.2
  },
  "text_profile": "CloudCorp provides cloud computing services and SaaS solutions to enterprises and businesses. We offer cloud infrastructure, platform services, and software applications for organizations across various industries.",
  "mode": "all_segments",
  "industry": "Technology"
}
```

### Example 3: Healthcare IT Company
```json
{
  "name": "HealthTech Solutions",
  "url": "https://healthtech.com",
  "products": ["Electronic Health Records", "Healthcare Software", "Patient Management Systems"],
  "customers": ["Hospitals", "Healthcare Providers", "Health Systems"],
  "segment_mix": {
    "Healthcare": 0.8,
    "Education": 0.1,
    "Commercial": 0.1
  },
  "text_profile": "HealthTech Solutions provides electronic health records and healthcare software to hospitals, healthcare providers, and health systems. We offer patient management systems and clinical software solutions.",
  "mode": "all_segments",
  "industry": "Healthcare"
}
```

### Example 4: Minimal Example (Required Fields Only)
```json
{
  "name": "My Company",
  "products": ["Payment Processing"],
  "customers": ["Banks"],
  "text_profile": "My Company offers payment processing services to banks."
}
```

## How to Create/Edit target.json

### Method 1: Manual Edit
1. Navigate to `comps/data/` directory
2. Create or edit `target.json` file
3. Use a JSON editor or text editor
4. Validate JSON syntax (no trailing commas, proper quotes)

### Method 2: Using Python Script
Create a Python script to generate target.json:

```python
import json

target = {
    "name": "Your Company Name",
    "url": "https://yourcompany.com",
    "products": ["Product 1", "Product 2"],
    "customers": ["Customer Type 1", "Customer Type 2"],
    "segment_mix": {
        "Healthcare": 0.5,
        "Education": 0.3,
        "Commercial": 0.2
    },
    "text_profile": "Your company description here...",
    "mode": "all_segments"
}

with open("comps/data/target.json", "w") as f:
    json.dump(target, f, indent=2)
```

### Method 3: Using a Form/Web Interface
You could create a simple web form to collect the data and generate the JSON file.

## Best Practices

1. **Products**
   - Use specific product names that match taxonomy.yaml aliases
   - Include 2-5 products for best results
   - Be consistent with naming (e.g., "Payment Processing" vs "Payments")

2. **Customers**
   - Use customer types that match taxonomy.yaml aliases
   - Include 2-5 customer types
   - Be specific (e.g., "Banks" not "Financial")

3. **Text Profile**
   - Write 2-3 comprehensive sentences
   - Include both products and customers
   - Be descriptive and specific
   - This is used for semantic similarity, so quality matters

4. **Segment Mix**
   - Use canonical segment names from taxonomy.yaml
   - Weights should sum to 1.0
   - If unsure, you can omit this field (defaults to neutral)

5. **Mode**
   - Must match a mode in `config/weights.yaml`
   - Default is `"all_segments"`
   - Check weights.yaml for available modes

## Validation

Before running the pipeline, validate your target.json:

```python
import json

# Load and validate
with open("comps/data/target.json", "r") as f:
    target = json.load(f)

# Check required fields
required = ["name", "products", "customers", "text_profile"]
for field in required:
    if field not in target:
        raise ValueError(f"Missing required field: {field}")

# Check products and customers are lists
assert isinstance(target["products"], list), "products must be a list"
assert isinstance(target["customers"], list), "customers must be a list"
assert len(target["products"]) > 0, "products must not be empty"
assert len(target["customers"]) > 0, "customers must not be empty"

# Check segment_mix if provided
if "segment_mix" in target:
    assert isinstance(target["segment_mix"], dict), "segment_mix must be a dict"
    total = sum(target["segment_mix"].values())
    assert abs(total - 1.0) < 0.01, f"segment_mix weights should sum to 1.0 (got {total})"

print("✓ target.json is valid!")
```

## Running the Pipeline

Once you have created `target.json`, run the pipeline:

```bash
cd comps
python cli/run_pipeline.py --target data/target.json
```

With OpenAI (for real embeddings and LLM extraction):

```bash
python cli/run_pipeline.py --target data/target.json --openai
```

## Output Files

The pipeline generates:
- `{target_id}_knn.csv` - KNN leaderboard (semantic similarity)
- `{target_id}_ranked.csv` - Final ranked comps (rule-based scores)
- `{target_id}_comps_meta.jsonl` - Metadata with concept matches and evidence
- `{target_id}_run_summary.json` - Run summary with provenance and metrics

Where `{target_id}` is derived from the `name` field (lowercased, spaces replaced with underscores).

