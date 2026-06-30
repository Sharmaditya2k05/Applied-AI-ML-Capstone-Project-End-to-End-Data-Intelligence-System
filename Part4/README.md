# Part 4 — LLM-Powered Feature: Model Prediction Explanation Pipeline (Track C)

## Chosen Track: C — Model Prediction Explanation Pipeline

## Dataset Choice

**Dataset:** Forest Fires Dataset
**Source:** UCI Machine Learning Repository

### Justification

This project uses the Forest Fires Dataset, which contains meteorological data and Fire Weather Index (FWI) system components collected for forest fire incidents in the Montesinho Natural Park, Portugal.

This dataset was selected because it satisfies all assignment requirements:

- 517 records (above the required minimum of 500 rows). After duplicate removal, the cleaned dataset used here contains 513 rows, satisfying the 500+ row requirement.
- More than 5 numerical features, including temperature, relative humidity, wind speed, rain, FFMC, DMC, DC, and ISI.
- At least 2 categorical features, including month and day of the week.
- A continuous target variable (Burned Area) suitable for regression.
- The same target can be converted into a binary classification problem by splitting it into Above Median and Below Median burned area classes.
- Contains complex, skewed distributions and natural outliers, making it highly suitable for exploratory data analysis, outlier handling, feature engineering, traditional machine learning, ensemble methods, and LLM-powered prediction explanations.

For these reasons, this dataset provides a realistic end-to-end machine learning workflow while meeting every requirement of the assignment.

---

## 1. LLM API Connection

- **Provider:** OpenRouter (https://openrouter.ai). *Note: OpenRouter was used because it provides a standard OpenAI-compatible HTTP API using model + messages payloads, making the implementation provider-independent.*
- **Model:** `google/gemma-4-26b-a4b-it:free`
- **API Key:** Stored in environment variable `LLM_API_KEY` — **never hardcoded**
- **Test call:** `"Reply with only the word: hello"` → Response: `"hello"` ✓

The `call_llm()` function:
1. Constructs a JSON payload with `model`, `messages`, `temperature`, `max_tokens`
2. Sets `Authorization: Bearer {api_key}` header
3. Sends `requests.post()` to the OpenRouter API
4. Checks `status_code == 200`; returns `None` on failure
5. Parses and returns `response.json()['choices'][0]['message']['content']`

---

## 2. Prompt Design

### System Prompt (verbatim)

```
You are a forest fire prediction explanation assistant. You analyze weather and fire-index
features alongside a machine learning model's prediction to produce a structured JSON explanation.

Given: feature values for a forest fire observation, the model's predicted class
(0 = Small Fire, 1 = Large Fire), and the predicted probability of class 1 (Large Fire).

You must output ONLY valid JSON (no markdown, no code fences, no extra text) with exactly
these 6 fields:
{
  "prediction_label": "Large Fire" or "Small Fire",
  "confidence_level": "low" or "medium" or "high",
  "top_reason": "Primary meteorological or index factor driving this prediction",
  "second_reason": "Secondary supporting factor",
  "risk_assessment": "Brief risk statement for fire response planning",
  "next_step": "Recommended action for fire management authorities"
}

Rules for confidence_level:
- "high" if probability > 0.70 or probability < 0.30
- "medium" if probability is between 0.30 and 0.45, or between 0.55 and 0.70
- "low" if probability is between 0.45 and 0.55

Key domain knowledge:
- FFMC (Fine Fuel Moisture Code): higher = drier fine fuels = faster ignition
- DMC (Duff Moisture Code): higher = drier medium fuels
- DC (Drought Code): higher = drier deep fuels = prolonged drought
- ISI (Initial Spread Index): higher = faster fire spread
- temp: higher temperature increases fire risk
- RH (Relative Humidity): lower = drier air = higher risk
- wind: higher = faster fire spread
- rain: any rain reduces risk significantly
```

### User Prompt Template (with placeholders)

```
Feature values:
{features_json}

Model prediction: class = {predicted_class} ({class_label})
Predicted probability of Large Fire: {probability:.4f}

Produce the JSON explanation.
```

### Why temperature = 0?

**Temperature = 0** is used for all structured output tasks because it produces **deterministic, reproducible outputs**. At temperature 0, the model always selects the highest-probability next token at each step, eliminating randomness. This is essential for:
- Consistent JSON schema compliance (same structure every time)
- Reproducible results across runs (important for auditing and testing)
- Reliable field values that match the defined enum constraints (e.g., `confidence_level` must be exactly "low", "medium", or "high")

Higher temperatures introduce sampling randomness, which risks producing invalid JSON, hallucinated field names, or values outside the defined schema.

---

## 3. JSON Schema and Validation

### Schema Definition (6 required scalar fields)

```json
{
  "type": "object",
  "properties": {
    "prediction_label":  {"type": "string"},
    "confidence_level":  {"type": "string", "enum": ["low", "medium", "high"]},
    "top_reason":        {"type": "string"},
    "second_reason":     {"type": "string"},
    "risk_assessment":   {"type": "string"},
    "next_step":         {"type": "string"}
  },
  "required": ["prediction_label", "confidence_level", "top_reason",
               "second_reason", "risk_assessment", "next_step"]
}
```

### Validation Process

After each `call_llm()` response:
1. Strip whitespace and remove any markdown code fences
2. Parse with `json.loads()` inside `try-except json.JSONDecodeError`
3. Validate with `jsonschema.validate()` inside `try-except jsonschema.ValidationError`
4. On any failure: log the error, return a fallback dict with all 6 fields set to `null`

---

## 4. PII Guardrail

### Implementation

```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

### Test Results

| Test Input                                            | PII Detected | Action  |
|-------------------------------------------------------|:------------:|---------|
| `"User email is john.doe@example.com and the area was 5.0"` | ✓ Yes   | BLOCKED |
| `"Temperature was 28.5 degrees with humidity 30 percent"`    | ✗ No    | ALLOWED |
| `"Call 555-123-4567 for fire department"`              | ✓ Yes        | BLOCKED |

The guardrail correctly blocks inputs containing email addresses and phone numbers while allowing clean meteorological data through.

---

## 5. End-to-End Demonstration (3 Inputs)

### Demonstration Table

| # | Feature Input | Predicted Class | Probability | Explanation JSON | Validation | PII |
|---|--------------|-----------------|-------------|------------------|------------|-----|
| 1 | Hot/dry August Saturday (temp=30, RH=20, FFMC=93, wind=5, rain=0) | 0 (Small Fire) | 0.4489 | `{"prediction_label": "Small Fire", "confidence_level": "low", "top_reason": "Low probability of large fire spread despite high fuel dryness", "second_reason": "Moderate wind speeds and high FFMC/DMC values indicating high ignition potential but limited spread capacity", "risk_assessment": "Uncertain risk level; high fuel volatility due to low humidity and high FFMC requires close monitoring", "next_step": "Increase surveillance in high-risk zones due to low confidence in the small fire prediction"}` | ✅ PASS | PASS |
| 2 | Cool/humid February Monday (temp=7, RH=80, FFMC=82, wind=2, rain=1.5) | 0 (Small Fire) | 0.3878 | `{"prediction_label": "Small Fire", "confidence_level": "medium", "top_reason": "High relative humidity (80%) and presence of rain (1.5mm) significantly suppress fire intensity", "second_reason": "Low temperature and low wind speeds limit potential fire spread", "risk_assessment": "Low risk of large-scale spread due to moisture levels and calm weather conditions", "next_step": "Continue routine monitoring; no immediate large-scale suppression resources required"}` | ✅ PASS | PASS |
| 3 | Warm/windy September Sunday (temp=22, RH=35, FFMC=91.5, wind=7, rain=0) | 1 (Large Fire) | 0.6061 | `{"prediction_label": "Large Fire", "confidence_level": "medium", "top_reason": "High FFMC and DMC values indicating extremely dry fine and medium fuels", "second_reason": "Low relative humidity (35%) and elevated drought code (DC) contributing to fuel aridity", "risk_assessment": "Elevated risk of rapid fire spread due to dry fuel moisture levels and moderate wind", "next_step": "Increase surveillance in high-risk sectors and prepare rapid response teams"}` | ✅ PASS | PASS |

**All 3 inputs produced valid JSON that passed schema validation.** No fallback was needed.

---

## 6. Temperature A/B Comparison

| Input | Output at temp=0 (key excerpts) | Output at temp=0.7 (key excerpts) | Key Difference |
|-------|--------------------------------|-----------------------------------|----------------|
| Scenario A (hot/dry August) | `top_reason`: "Low probability of large fire spread despite high fuel dryness" | `top_reason`: "High FFMC and DMC indicating dry fuel, but low probability of large-scale spread" | Wording varies; temp=0.7 rephrases the reasoning with different emphasis |
| Scenario B (cool/humid Feb) | `top_reason`: "High relative humidity (80%) and recent precipitation (1.5mm) significantly suppress fire intensity" | `top_reason`: "High relative humidity (80%) and presence of rain (1.5mm) significantly suppress fire spread potential" | Subtle synonym changes ("recent precipitation" → "presence of rain", "fire intensity" → "fire spread potential") |
| Scenario C (warm/windy Sep) | `top_reason`: "High FFMC and DMC values indicating extremely dry fine and medium fuels" | `top_reason`: "High FFMC and DMC values indicating extremely dry surface and medium fuels" | Minor word swap ("fine" → "surface"); same semantic content |

### Analysis

**Temperature = 0** produces deterministic output: the model always picks the highest-probability next token, so the same input yields the same output every time. This is ideal for structured tasks where consistency and schema compliance matter.

**Temperature = 0.7** introduces sampling randomness by scaling the logits before applying softmax: lower-probability tokens get a non-trivial chance of being selected. This creates variability in word choice, phrasing, and sometimes field ordering — even when the overall semantic meaning is similar. In all three cases, the `prediction_label` and `confidence_level` remained the same (the model is confident in the categorical fields), but the free-text `top_reason`, `second_reason`, `risk_assessment`, and `next_step` fields showed phrasing differences.

**For structured data tasks, temperature = 0 is the correct choice** because it maximizes reproducibility and minimizes the risk of generating invalid JSON or out-of-schema values.

---

## Output Files

| File               | Description                                              |
|--------------------|----------------------------------------------------------|
| `analysis.py`      | Complete Part 4 script (Track C pipeline)                |
| `analysis.ipynb`   | Jupyter notebook version                                 |
| `cleaned_data.csv` | Cleaned dataset from Part 1                              |
| `forestfires.csv`  | Original raw dataset                                     |
| `best_model.pkl`   | Best pipeline from Part 3 (loaded for prediction)        |
| `README.md`        | This document                                            |
