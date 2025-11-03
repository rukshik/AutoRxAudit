# AutoRxAudit API

AI-powered opioid prescription auditing system using deep learning models.

## Architecture

```
POST /audit-prescription
  ↓
Query PostgreSQL for patient features
  ↓
Run Two Models:
  • Eligibility (DNN 50K) - Clinical need assessment
  • OUD Risk (DNN 10K) - Opioid Use Disorder risk
  ↓
Flag if: NOT Eligible OR High OUD Risk
  ↓
Return audit decision + explanation
```

## Setup

### 1. Install Dependencies

```bash
pip install fastapi uvicorn psycopg2-binary torch numpy pandas
```

### 2. Configure PostgreSQL

Create database:
```sql
CREATE DATABASE autorxaudit;
```

Set environment variables (or use defaults):
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=autorxaudit
export DB_USER=postgres
export DB_PASSWORD=postgres
```

### 3. Create Schema and Populate Data

```bash
cd api/database
python populate_db.py
```

This will:
- Create `patients` and `audit_logs` tables
- Insert 500 synthetic patients from test dataset
- Display statistics

### 4. Run API Server

```bash
cd api
python app.py
# Or use uvicorn directly:
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API available at: `http://localhost:8000`

## API Endpoints

### `POST /audit-prescription`

Audit an opioid prescription request.

**Request:**
```json
{
  "patient_id": "PAT_00001",
  "prescription_id": "RX_12345",
  "drug_name": "Oxycodone 5mg"
}
```

**Response:**
```json
{
  "flagged": true,
  "patient_id": "PAT_00001",
  "prescription_id": "RX_12345",
  "eligibility_score": 0.35,
  "eligibility_prediction": 0,
  "oud_risk_score": 0.82,
  "oud_risk_prediction": 1,
  "flag_reason": "No clinical need for opioids (eligibility: 35.0%) AND High OUD risk detected (risk: 82.0%)",
  "recommendation": "REVIEW REQUIRED: Consider alternative pain management or additional evaluation",
  "audited_at": "2025-11-02T14:30:00"
}
```

### `GET /patients/{patient_id}`

Get patient feature data.

**Response:**
```json
{
  "patient_id": "PAT_00001",
  "features": {
    "age_at_first_admit": 45.2,
    "bmi": 28.5,
    "n_hospital_admits": 3,
    "opioid_rx_count": 12,
    ...
  },
  "created_at": "2025-11-02T10:00:00"
}
```

### `GET /audit-history/{patient_id}`

Get audit history for a patient.

**Response:**
```json
{
  "patient_id": "PAT_00001",
  "total_audits": 5,
  "audits": [
    {
      "audit_id": 123,
      "flagged": true,
      "eligibility_score": 0.35,
      "oud_risk_score": 0.82,
      "flag_reason": "...",
      "audited_at": "2025-11-02T14:30:00"
    },
    ...
  ]
}
```

### `GET /stats`

Get overall audit statistics.

**Response:**
```json
{
  "total_audits": 150,
  "flagged_prescriptions": 45,
  "flag_rate": 0.30,
  "avg_eligibility_score": 0.67,
  "avg_oud_risk_score": 0.15
}
```

### `GET /`

Health check endpoint.

## Models

### Eligibility Model (DNN 50K)
- **Purpose:** Assess clinical need for opioids
- **Features:** 16 clinical features (NO opioid history)
- **Accuracy:** 81.94%
- **Dataset:** 50,000 patients

### OUD Risk Model (DNN 10K)
- **Purpose:** Predict Opioid Use Disorder risk
- **Features:** 19 features (includes opioid history)
- **Accuracy:** 99.87%
- **Dataset:** 10,000 patients

## Audit Logic

Prescription is **FLAGGED** if:
```python
(Eligibility == NOT_ELIGIBLE) OR (OUD_Risk == HIGH)
```

- **NOT_ELIGIBLE:** Patient doesn't have sufficient clinical need
- **HIGH_OUD_RISK:** Patient shows high risk for opioid use disorder

## Testing

### Interactive API Documentation

Visit `http://localhost:8000/docs` for Swagger UI

### Test with cURL

```bash
# Audit prescription
curl -X POST http://localhost:8000/audit-prescription \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT_00001",
    "prescription_id": "RX_TEST_001",
    "drug_name": "Oxycodone 10mg"
  }'

# Get patient data
curl http://localhost:8000/patients/PAT_00001

# Get audit history
curl http://localhost:8000/audit-history/PAT_00001

# Get statistics
curl http://localhost:8000/stats
```

### Test with Python

```python
import requests

# Audit prescription
response = requests.post(
    "http://localhost:8000/audit-prescription",
    json={
        "patient_id": "PAT_00001",
        "prescription_id": "RX_12345",
        "drug_name": "Hydrocodone 5mg"
    }
)

result = response.json()
print(f"Flagged: {result['flagged']}")
print(f"Reason: {result['flag_reason']}")
print(f"Recommendation: {result['recommendation']}")
```

## Database Schema

### Patients Table
Stores patient clinical features for model inference.

Key fields:
- Demographics: age, BMI
- Hospital history: admissions, ICU stays, length of stay
- Severity indicators: DRG severity, mortality risk
- Medication history: ATC codes
- Opioid exposure: prescriptions, days exposed (OUD model only)

### Audit Logs Table
Stores all audit decisions for tracking and analysis.

Fields:
- Patient and prescription IDs
- Model predictions and scores
- Flag status and reason
- Timestamp

## Production Considerations

### Security
- [ ] Add authentication (JWT tokens)
- [ ] Encrypt sensitive patient data
- [ ] Rate limiting
- [ ] Input validation and sanitization

### Performance
- [ ] Model caching (already done - loaded at startup)
- [ ] Database connection pooling
- [ ] Async database operations
- [ ] Load balancing for high volume

### Monitoring
- [ ] Logging (structured logging)
- [ ] Metrics (Prometheus)
- [ ] Alerting (flagging rate anomalies)
- [ ] Model performance tracking

### Compliance
- [ ] HIPAA compliance audit
- [ ] Audit trail retention policy
- [ ] Data access logging
- [ ] Regular model retraining schedule

## License

See main project LICENSE file.
