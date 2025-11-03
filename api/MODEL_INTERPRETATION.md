# Model Interpretation Guide

## Understanding the Two Models

### 1. Eligibility Model
**What it predicts:** `opioid_eligibility` 
- **Label = 1**: Patient HAS clinical need for opioids (e.g., has pain diagnosis)
- **Label = 0**: Patient does NOT have clinical need for opioids

**Interpretation:**
- **Score >= 0.5** → Prediction = 1 → Patient IS eligible
- **Score < 0.5** → Prediction = 0 → Patient NOT eligible

**Example from validation:**
```
Patient 20038695: Eligibility score = 0.898 (90%) → pred = 1 → ELIGIBLE ✅
Patient 20038478: Eligibility score = 0.163 (16%) → pred = 0 → NOT ELIGIBLE 🚩
```

### 2. OUD Risk Model
**What it predicts:** `y_oud` (Opioid Use Disorder risk)
- **Label = 1**: Patient HAS high OUD risk (addiction risk)
- **Label = 0**: Patient does NOT have high OUD risk

**Interpretation:**
- **Score >= 0.5** → Prediction = 1 → HIGH OUD risk
- **Score < 0.5** → Prediction = 0 → LOW OUD risk

**Example from validation:**
```
Patient 20038695: OUD score = 0.104 (10%) → pred = 0 → LOW RISK ✅
Patient 20002189: OUD score = 0.184 (18%) → pred = 0 → LOW RISK (but higher than 10%)
```

---

## Decision Logic

### Flagging Rules:
```python
not_eligible = (eligibility_pred == 0)  # No clinical need
high_oud_risk = (oud_pred == 1)        # High addiction risk
flagged = not_eligible OR high_oud_risk
```

### Decision Matrix:

| Eligibility | OUD Risk | Result | Reason |
|------------|----------|--------|--------|
| pred=1 (eligible) | pred=0 (low risk) | ✅ **APPROVED** | Has clinical need + low addiction risk |
| pred=1 (eligible) | pred=1 (high risk) | 🚩 **FLAGGED** | Has clinical need BUT high addiction risk → needs review |
| pred=0 (not eligible) | pred=0 (low risk) | 🚩 **FLAGGED** | No clinical need → inappropriate prescription |
| pred=0 (not eligible) | pred=1 (high risk) | 🚩 **FLAGGED** | No clinical need AND high addiction risk |

---

## Answering Your Question

**Q: "Both Eligibility and OUD Risk are on same directional scale?"**

**A: NO - They are OPPOSITE directional scales:**

### Eligibility Model (POSITIVE is GOOD):
- **High score (>0.5)** = GOOD → Patient eligible → helps APPROVAL
- **Low score (<0.5)** = BAD → Patient not eligible → triggers FLAG

### OUD Risk Model (NEGATIVE is GOOD):
- **Low score (<0.5)** = GOOD → Low addiction risk → helps APPROVAL  
- **High score (>0.5)** = BAD → High addiction risk → triggers FLAG

### Summary:
```
For APPROVAL, you need:
  - Eligibility HIGH (pred=1, score >= 0.5) ✅
  AND
  - OUD Risk LOW (pred=0, score < 0.5) ✅

Any other combination = FLAGGED 🚩
```

---

## Validation Results Explained

### Eligible Patients (APPROVED):
```
Patient 20038695:
  Eligibility: 0.898 (HIGH ✅) → pred=1 → eligible
  OUD Risk:    0.104 (LOW ✅)  → pred=0 → low risk
  Result: APPROVED (both conditions met)

Patient 20033109:
  Eligibility: 0.636 (HIGH ✅) → pred=1 → eligible
  OUD Risk:    0.086 (LOW ✅)  → pred=0 → low risk
  Result: APPROVED (both conditions met)
```

### Ineligible Patients (FLAGGED):
```
Patient 20038478:
  Eligibility: 0.163 (LOW 🚩) → pred=0 → not eligible
  OUD Risk:    0.112 (LOW ✅) → pred=0 → low risk
  Result: FLAGGED (no clinical need)

Patient 20002189:
  Eligibility: 0.170 (LOW 🚩) → pred=0 → not eligible
  OUD Risk:    0.184 (LOW ✅) → pred=0 → low risk
  Result: FLAGGED (no clinical need)
```

---

## Threshold: Is 0.5 the decision point?

**YES**, 0.5 is the threshold for binary classification:

```python
# In run_inference() function:
prob = 1 / (1 + np.exp(-logits))  # Convert logits to probability
prediction = 1 if prob >= 0.5 else 0  # Binary decision at 0.5
```

- **Eligibility**: Score >= 0.5 → pred=1 (eligible)
- **OUD Risk**: Score >= 0.5 → pred=1 (high risk)

The 0.5 threshold is standard for binary classification with balanced decision boundaries.
