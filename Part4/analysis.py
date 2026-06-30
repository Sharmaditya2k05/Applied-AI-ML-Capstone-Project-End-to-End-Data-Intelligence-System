"""
Part 4 — LLM-Powered Feature: Model Prediction Explanation Pipeline (Track C)
Forest Fires Dataset
"""

import warnings
warnings.filterwarnings("ignore")

import os
import re
import json
import requests
import numpy as np
import pandas as pd
import joblib
import jsonschema

SEP = "\n" + "=" * 80 + "\n"

# =============================================================================
# TASK 1 — Set up LLM API connection
# =============================================================================
print(SEP + "TASK 1: LLM API Connection Setup" + SEP)

# API key from environment variable — NEVER hardcoded
api_key = os.environ.get("LLM_API_KEY", "")
if not api_key:
    print("⚠ WARNING: LLM_API_KEY environment variable not set.")
    print("  Set it with: export LLM_API_KEY='your-api-key-here'")
    print("  Using OpenRouter: https://openrouter.ai/keys")

# Configurable model and URL (defaults to OpenRouter)
LLM_URL = os.environ.get("LLM_URL", "https://openrouter.ai/api/v1/chat/completions")
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-4-26b-a4b-it:free")

print(f"API URL:  {LLM_URL}")
print(f"Model:    {LLM_MODEL}")
print(f"API Key:  {'***' + api_key[-4:] if len(api_key) > 4 else '(not set)'}")


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    """
    Reusable function to call the LLM API.
    Returns the model's text response, or None on failure.
    """
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(LLM_URL, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            print(f"  ✗ API error: status {response.status_code}")
            print(f"    Response: {response.text[:300]}")
            return None
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Request error: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"  ✗ Parse error: {e}")
        return None


# Test with a simple prompt
print("\n--- Test Call ---")
test_response = call_llm(
    system_prompt="You are a helpful assistant.",
    user_prompt="Reply with only the word: hello",
    temperature=0.0,
    max_tokens=10,
)
print(f"Test response: {test_response}")
if test_response:
    print("✓ LLM API connection working.")
else:
    print("⚠ LLM API connection failed. Continuing with demonstration structure.")

# =============================================================================
# PII GUARDRAIL
# =============================================================================
print(SEP + "PII GUARDRAIL SETUP" + SEP)


def has_pii(text):
    """Check if text contains personally identifiable information (email or phone)."""
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


def safe_call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    """Wrapper that checks for PII before calling the LLM."""
    if has_pii(user_prompt):
        print("  ✗ Input blocked: PII detected.")
        return None
    return call_llm(system_prompt, user_prompt, temperature, max_tokens)


# Demonstrate PII guardrail
print("--- PII Guardrail Tests ---")

pii_test_1 = "User email is john.doe@example.com and the area was 5.0"
pii_test_2 = "Temperature was 28.5 degrees with humidity 30 percent"
pii_test_3 = "Call 555-123-4567 for fire department"

print(f"\nTest 1: '{pii_test_1}'")
print(f"  has_pii = {has_pii(pii_test_1)} → {'BLOCKED' if has_pii(pii_test_1) else 'ALLOWED'}")

print(f"\nTest 2: '{pii_test_2}'")
print(f"  has_pii = {has_pii(pii_test_2)} → {'BLOCKED' if has_pii(pii_test_2) else 'ALLOWED'}")

print(f"\nTest 3: '{pii_test_3}'")
print(f"  has_pii = {has_pii(pii_test_3)} → {'BLOCKED' if has_pii(pii_test_3) else 'ALLOWED'}")

# =============================================================================
# LOAD MODEL FROM PART 3
# =============================================================================
print(SEP + "LOAD BEST MODEL FROM PART 3" + SEP)

loaded_model = joblib.load("best_model.pkl")
print(f"✓ Loaded best_model.pkl")
print(f"  Pipeline steps: {[step[0] for step in loaded_model.steps]}")

# =============================================================================
# DEFINE SCHEMA FOR EXPLANATION OUTPUT
# =============================================================================

explanation_schema = {
    "type": "object",
    "properties": {
        "prediction_label": {
            "type": "string",
            "description": "Human-readable label: 'Large Fire' or 'Small Fire'"
        },
        "confidence_level": {
            "type": "string",
            "enum": ["low", "medium", "high"],
            "description": "Confidence in the prediction"
        },
        "top_reason": {
            "type": "string",
            "description": "Primary factor driving this prediction"
        },
        "second_reason": {
            "type": "string",
            "description": "Secondary factor supporting the prediction"
        },
        "risk_assessment": {
            "type": "string",
            "description": "Brief risk assessment for fire response planning"
        },
        "next_step": {
            "type": "string",
            "description": "Recommended next action for fire management"
        }
    },
    "required": [
        "prediction_label", "confidence_level",
        "top_reason", "second_reason",
        "risk_assessment", "next_step"
    ]
}

print("Explanation JSON schema defined with 6 required scalar fields:")
for field in explanation_schema["required"]:
    ftype = explanation_schema["properties"][field].get("type", "string")
    print(f"  • {field}: {ftype}")

# =============================================================================
# DEFINE FEATURE ENCODING FUNCTION
# =============================================================================


def encode_record(features_dict):
    """
    Convert a raw feature dictionary into a DataFrame matching the model's
    expected input format (same encoding as Part 2/3).
    """
    df = pd.DataFrame([features_dict])

    # Label-encode month
    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                 "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    if "month" in df.columns and isinstance(df["month"].iloc[0], str):
        df["month"] = df["month"].map(month_map)

    # One-hot encode day
    all_days = ["fri", "mon", "sat", "sun", "thu", "tue", "wed"]
    if "day" in df.columns:
        day_val = df["day"].iloc[0]
        df = df.drop(columns=["day"])
        for d in all_days:
            if d == "fri":  # drop_first reference
                continue
            df[f"day_{d}"] = 1 if day_val == d else 0

    # Ensure correct column order
    expected_cols = ["X", "Y", "month", "FFMC", "DMC", "DC", "ISI",
                     "temp", "RH", "wind", "rain",
                     "day_mon", "day_sat", "day_sun", "day_thu", "day_tue", "day_wed"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_cols]
    return df


# =============================================================================
# DEFINE 3 HAND-CRAFTED TEST INPUTS
# =============================================================================

test_inputs = [
    {
        "name": "Scenario A: Hot dry August Saturday",
        "features": {
            "X": 7, "Y": 5, "month": "aug", "day": "sat",
            "FFMC": 93.0, "DMC": 150.0, "DC": 700.0, "ISI": 15.0,
            "temp": 30.0, "RH": 20, "wind": 5.0, "rain": 0.0,
        }
    },
    {
        "name": "Scenario B: Cool humid February Monday",
        "features": {
            "X": 3, "Y": 4, "month": "feb", "day": "mon",
            "FFMC": 82.0, "DMC": 25.0, "DC": 80.0, "ISI": 4.0,
            "temp": 7.0, "RH": 80, "wind": 2.0, "rain": 1.5,
        }
    },
    {
        "name": "Scenario C: Warm windy September Sunday",
        "features": {
            "X": 6, "Y": 5, "month": "sep", "day": "sun",
            "FFMC": 91.5, "DMC": 130.0, "DC": 650.0, "ISI": 11.0,
            "temp": 22.0, "RH": 35, "wind": 7.0, "rain": 0.0,
        }
    },
]

# =============================================================================
# SYSTEM PROMPT (Track C — zero-shot)
# =============================================================================

SYSTEM_PROMPT = """You are a forest fire prediction explanation assistant. You analyze weather and fire-index features alongside a machine learning model's prediction to produce a structured JSON explanation.

Given: feature values for a forest fire observation, the model's predicted class (0 = Small Fire, 1 = Large Fire), and the predicted probability of class 1 (Large Fire).

You must output ONLY valid JSON (no markdown, no code fences, no extra text) with exactly these 6 fields:
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
- rain: any rain reduces risk significantly"""

USER_PROMPT_TEMPLATE = """Feature values:
{features_json}

Model prediction: class = {predicted_class} ({class_label})
Predicted probability of Large Fire: {probability:.4f}

Produce the JSON explanation."""

# =============================================================================
# TASK 2–5: END-TO-END PIPELINE (3 inputs)
# =============================================================================
print(SEP + "TRACK C: MODEL PREDICTION EXPLANATION PIPELINE" + SEP)

FALLBACK = {
    "prediction_label": None,
    "confidence_level": None,
    "top_reason": None,
    "second_reason": None,
    "risk_assessment": None,
    "next_step": None,
}

results = []

for i, test_input in enumerate(test_inputs, 1):
    print(f"\n{'─' * 70}")
    print(f"INPUT {i}: {test_input['name']}")
    print(f"{'─' * 70}")

    features = test_input["features"]
    encoded = encode_record(features)

    # Get model prediction
    predicted_class = loaded_model.predict(encoded)[0]
    predicted_proba = loaded_model.predict_proba(encoded)[0, 1]
    class_label = "Large Fire" if predicted_class == 1 else "Small Fire"

    print(f"  Features: {json.dumps(features, indent=2)}")
    print(f"  Predicted class: {predicted_class} ({class_label})")
    print(f"  P(Large Fire):   {predicted_proba:.4f}")

    # Construct user prompt
    features_json = json.dumps(features, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        features_json=features_json,
        predicted_class=predicted_class,
        class_label=class_label,
        probability=predicted_proba,
    )

    # PII check + LLM call
    pii_status = "BLOCKED" if has_pii(user_prompt) else "PASS"
    print(f"\n  PII check: {pii_status}")
    raw_response = safe_call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=512)

    # Parse and validate
    validation_status = "fail"
    explanation = FALLBACK.copy()

    if raw_response:
        print(f"\n  Raw LLM response:\n  {raw_response[:500]}")
        # Strip markdown code fences if present
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)
            try:
                jsonschema.validate(instance=parsed, schema=explanation_schema)
                validation_status = "pass"
                explanation = parsed
                print(f"\n  ✓ Schema validation: PASS")
            except jsonschema.ValidationError as e:
                print(f"\n  ✗ Schema validation FAILED: {e.message}")
                explanation = FALLBACK.copy()
        except json.JSONDecodeError as e:
            print(f"\n  ✗ JSON parsing FAILED: {e}")
            explanation = FALLBACK.copy()
    else:
        print(f"\n  ✗ No response from LLM (API error or PII block)")

    results.append({
        "input_name": test_input["name"],
        "features": features,
        "predicted_class": predicted_class,
        "class_label": class_label,
        "probability": predicted_proba,
        "raw_response": raw_response,
        "explanation": explanation,
        "validation": validation_status,
        "pii_status": pii_status,
    })

    print(f"\n  Final explanation: {json.dumps(explanation, indent=2)}")
    print(f"  Validation status: {validation_status.upper()}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print(SEP + "DEMONSTRATION SUMMARY TABLE" + SEP)

print(f"{'#':<3} {'Scenario':<40} {'Class':>6} {'Prob':>7} {'Valid':>6} {'PII':>7}")
print("-" * 72)
for i, r in enumerate(results, 1):
    print(f"{i:<3} {r['input_name']:<40} {r['predicted_class']:>6} "
          f"{r['probability']:>7.4f} {r['validation']:>6} {r['pii_status']:>7}")

# =============================================================================
# TEMPERATURE A/B COMPARISON
# =============================================================================
print(SEP + "TEMPERATURE A/B COMPARISON (temp=0 vs temp=0.7)" + SEP)

temp_comparison = []
for i, test_input in enumerate(test_inputs, 1):
    features = test_input["features"]
    encoded = encode_record(features)
    predicted_class = loaded_model.predict(encoded)[0]
    predicted_proba = loaded_model.predict_proba(encoded)[0, 1]
    class_label = "Large Fire" if predicted_class == 1 else "Small Fire"

    features_json = json.dumps(features, indent=2)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        features_json=features_json,
        predicted_class=predicted_class,
        class_label=class_label,
        probability=predicted_proba,
    )

    print(f"\n--- Input {i}: {test_input['name']} ---")

    # Temperature = 0
    resp_t0 = safe_call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=512)
    print(f"  temp=0.0 response: {resp_t0[:200] if resp_t0 else 'None'}...")

    # Temperature = 0.7
    resp_t07 = safe_call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7, max_tokens=512)
    print(f"  temp=0.7 response: {resp_t07[:200] if resp_t07 else 'None'}...")

    temp_comparison.append({
        "input": test_input["name"],
        "resp_t0": resp_t0,
        "resp_t07": resp_t07,
    })

print("\n\n--- Temperature Comparison Summary ---")
for i, tc in enumerate(temp_comparison, 1):
    print(f"\nInput {i}: {tc['input']}")
    if tc["resp_t0"] and tc["resp_t07"]:
        same = tc["resp_t0"].strip() == tc["resp_t07"].strip()
        print(f"  Outputs identical: {same}")
        if not same:
            # Try to parse both and compare specific fields
            try:
                j0 = json.loads(tc["resp_t0"].strip().replace("```json", "").replace("```", ""))
                j7 = json.loads(tc["resp_t07"].strip().replace("```json", "").replace("```", ""))
                diffs = [k for k in j0 if k in j7 and j0[k] != j7[k]]
                print(f"  Fields that differ: {diffs}")
            except Exception:
                print("  (Could not parse for field-level comparison)")
    else:
        print(f"  temp=0: {'received' if tc['resp_t0'] else 'no response'}")
        print(f"  temp=0.7: {'received' if tc['resp_t07'] else 'no response'}")

print(SEP + "PART 4 COMPLETE — All tasks executed successfully." + SEP)
