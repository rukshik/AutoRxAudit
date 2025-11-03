# AutoRxAudit - Setup & Usage Guide

Complete guide for running the AutoRxAudit prescription auditing web application.

## Overview

AutoRxAudit is an AI-powered prescription auditing system with:
- **Dual AI Models**: Eligibility (81.94% AUC) + OUD Risk (99.87% AUC)
- **React Frontend**: Login, audit form, results display, action recording
- **FastAPI Backend**: REST API with database integration
- **PostgreSQL Database**: MIMIC-IV demo data + user authentication

---

## Prerequisites

✅ **Backend API** (already completed in previous session)
✅ **Database** with MIMIC-IV demo data
✅ **React Frontend** (just created)

### Required Software:
- Python 3.8+
- Node.js 14+
- PostgreSQL access to Azure database

---

## Setup Instructions

### Step 1: Apply Database Schema

Create the `users` and `audit_actions` tables:

```bash
cd api/database
psql -h autorxaudit-server.postgres.database.azure.com -U cloudsa -d mimiciv_demo_raw -f schema_with_users.sql
```

**What this does:**
- Creates `users` table with 3 demo accounts
- Creates `audit_actions` table for tracking decisions
- Sample users: doctor, pharmacist, admin (all with password123)

### Step 2: Start Backend API

```bash
cd api
python -m uvicorn app:app --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

**Test the API:**
```bash
# In another terminal:
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

### Step 3: Start Frontend

```bash
cd frontend
npm install  # First time only
npm start
```

**Expected output:**
```
Compiled successfully!

You can now view frontend in the browser.

  Local:            http://localhost:3000
```

Browser will automatically open to `http://localhost:3000`

---

## Demo Accounts

| Email | Password | Role | Notes |
|-------|----------|------|-------|
| doctor@hospital.com | password123 | Doctor | Full access |
| pharmacist@hospital.com | password123 | Pharmacist | Full access |
| admin@hospital.com | admin123 | Admin | Full access |

---

## Complete Usage Walkthrough

### 1. Login

1. Open `http://localhost:3000`
2. Enter email: `doctor@hospital.com`
3. Enter password: `password123`
4. Click "Login"

**What happens:**
- Frontend sends POST to `/api/login`
- Backend validates plain password match
- Returns user object with session_id
- Frontend stores user in localStorage

### 2. Audit Prescription

1. **Select Patient**: Choose from dropdown (e.g., `20038695`)
2. **Select Drug**: Choose from dropdown (e.g., `Oxycodone` or `Ibuprofen`)
3. Click **"Audit Prescription"**

**What happens:**
- Frontend sends POST to `/audit-prescription`
- Backend:
  - Extracts patient features from database
  - Calculates 43 engineered features (age, conditions, medications, etc.)
  - Runs Eligibility model (predicts clinical need)
  - Runs OUD Risk model (predicts addiction risk)
  - Applies business rules to flag prescriptions
- Frontend displays results

### 3. Review Results

**Eligibility Score (0-100%):**
- Higher = Patient has clinical need (pain diagnosis)
- Prediction: ELIGIBLE or NOT ELIGIBLE

**OUD Risk Score (0-100%):**
- Higher = Higher risk of opioid use disorder
- Prediction: HIGH RISK or LOW RISK

**Flag Status:**
- 🚩 **FLAGGED**: One of these conditions:
  - Not eligible + prescribing opioid
  - High OUD risk + prescribing opioid
  - Eligible + NOT prescribing opioid
- ✅ **APPROVED**: Safe to prescribe

**Recommendation:**
- Plain English explanation of flag reason
- Suggested action

### 4. Take Action

**If Flagged:**
- **Deny Prescription**: Agree with AI recommendation
- **Override & Approve**: Disagree with AI, approve anyway

**If Approved:**
- **Approve Prescription**: Agree with AI recommendation
- **Override & Deny**: Disagree with AI, deny anyway

**Optional**: Add reason for decision in text box

Click the action button → Frontend sends POST to `/api/audit-action`

**What happens:**
- Backend records action in `audit_actions` table
- Stores: patient_id, drug_name, scores, action, reason, timestamp, user
- Frontend shows success message
- Form resets for next audit

### 5. View Audit History

Click **"View Audit History"** button

**What happens:**
- Frontend sends GET to `/api/audit-history`
- Backend returns all audit records from database
- Frontend displays table with columns:
  - Date & Time
  - Patient ID
  - Drug
  - Eligibility score
  - OUD Risk score
  - Flagged status
  - Action taken
  - Clinician name
  - Reason (if provided)

### 6. Logout

Click **"Logout"** button
- Clears localStorage
- Returns to login page

---

## Project Structure

```
AutoRxAudit/
├── api/
│   ├── app.py                          # FastAPI backend (750+ lines)
│   ├── requirements.txt                # Python dependencies
│   ├── database/
│   │   ├── schema_raw.sql              # Original MIMIC-IV schema
│   │   └── schema_with_users.sql       # Extended with users/audit_actions
│   └── models/
│       ├── eligibility_model.pkl       # Eligibility classifier (AUC 81.94%)
│       └── oud_model.pkl               # OUD risk classifier (AUC 99.87%)
│
├── frontend/
│   ├── src/
│   │   ├── App.js                      # Main app with routing
│   │   ├── Login.js                    # Login component
│   │   ├── Login.css                   # Login styling
│   │   ├── PrescriptionForm.js         # Main audit form
│   │   ├── PrescriptionForm.css        # Form styling
│   │   ├── AuditHistory.js             # History table
│   │   └── AuditHistory.css            # History styling
│   ├── public/
│   └── package.json
│
├── data/
│   └── mimic-clinical-iv-demo/         # MIMIC-IV demo dataset
│
└── README.md                            # This file
```

---

## API Endpoints Reference

### Authentication
- `POST /api/login` - Login with email/password
  - Body: `{"email": "doctor@hospital.com", "password": "password123"}`
  - Returns: `{"success": true, "user": {...}}`

- `GET /api/me` - Get current user info
  - Returns: `{"user_id": 1, "email": "...", "full_name": "...", "role": "..."}`

### Data Endpoints
- `GET /api/patients` - Get patient list for dropdown
  - Returns: `[{"patient_id": "20038695", "gender": "M", "date_of_birth": "..."}, ...]`

- `GET /api/drugs` - Get drug list
  - Returns: `{"opioids": [...], "non_opioids": [...]}`

### Audit Endpoints
- `POST /audit-prescription` - AI model inference
  - Body: `{"patient_id": "20038695", "drug_name": "Oxycodone"}`
  - Returns: Full audit result with scores, predictions, flag status

- `POST /api/audit-action` - Record user decision
  - Body: `{"patient_id": "...", "drug_name": "...", "action": "APPROVED", ...}`
  - Returns: `{"success": true, "audit_id": 1}`

- `GET /api/audit-history` - Get audit records
  - Returns: `[{"audit_id": 1, "patient_id": "...", "action": "...", ...}, ...]`

### Health Check
- `GET /health` - Server health status
  - Returns: `{"status": "healthy"}`

---

## Testing Scenarios

### Scenario 1: Appropriate Opioid Prescription
1. Login as doctor
2. Patient: `20038695` (has pain diagnosis)
3. Drug: `Oxycodone`
4. Expected: **✅ APPROVED** (eligible + low OUD risk)
5. Action: Click "Approve Prescription"

### Scenario 2: Inappropriate Opioid Prescription
1. Login as doctor
2. Patient: Select patient without pain diagnosis
3. Drug: `Oxycodone`
4. Expected: **🚩 FLAGGED** (not eligible + prescribing opioid)
5. Action: Click "Deny Prescription" OR "Override & Approve" with reason

### Scenario 3: High Risk Patient
1. Login as doctor
2. Patient: Select patient with substance abuse history
3. Drug: `Oxycodone`
4. Expected: **🚩 FLAGGED** (high OUD risk)
5. Action: Consider alternative medication or click "Override & Approve" with reason

### Scenario 4: Non-Opioid Medication
1. Login as pharmacist
2. Patient: Any patient
3. Drug: `Ibuprofen`
4. Expected: Varies based on patient eligibility
5. Action: Review and approve/deny

---

## Troubleshooting

### Frontend won't start
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm start
```

### Backend errors: "Module not found"
```bash
cd api
pip install -r requirements.txt
```

### Database connection failed
Check `.env` file or connection string in `app.py`:
```python
DATABASE_URL = "postgresql://cloudsa@autorxaudit-server:password@autorxaudit-server.postgres.database.azure.com:5432/mimiciv_demo_raw"
```

### Login fails: "Invalid credentials"
Make sure you ran `schema_with_users.sql`:
```bash
cd api/database
psql -h autorxaudit-server.postgres.database.azure.com -U cloudsa -d mimiciv_demo_raw -f schema_with_users.sql
```

### CORS errors in browser console
Backend should have CORS enabled:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Prescription audit returns error
Check that:
1. Patient exists in database
2. Backend models are loaded (check terminal for loading messages)
3. Feature extraction queries are working

---

## Security Notes

⚠️ **This is a DEVELOPMENT version with simplified security:**

- Plain text passwords (no bcrypt hashing)
- No JWT tokens
- Session stored in localStorage
- No HTTPS
- CORS allows all origins

**NOT SUITABLE FOR PRODUCTION**

### For Production Deployment:

1. **Password Security**: Use bcrypt or similar hashing
2. **Token-based Auth**: Implement JWT with expiration
3. **HTTPS**: Enable SSL/TLS
4. **CORS**: Restrict to specific domains
5. **Environment Variables**: Move secrets to .env
6. **Session Management**: Use secure, httpOnly cookies
7. **Input Validation**: Add comprehensive validation
8. **Rate Limiting**: Prevent abuse
9. **Logging**: Audit all actions
10. **Monitoring**: Add health checks and alerts

---

## Next Steps

### Enhancements to Consider:

1. **Search/Filter**: Add search in audit history
2. **Pagination**: Limit audit history results
3. **Export**: Download audit history as CSV
4. **Charts**: Visualize audit statistics
5. **Notifications**: Email alerts for flagged prescriptions
6. **Mobile**: Responsive design improvements
7. **Dark Mode**: Theme toggle
8. **Profile**: User profile page
9. **Bulk Upload**: Upload multiple prescriptions
10. **Reports**: Monthly audit summary reports

---

## Support

For issues or questions:
1. Check console logs (browser DevTools + terminal)
2. Verify all prerequisites are installed
3. Ensure database connection is working
4. Test API endpoints independently with curl/Postman

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- **MIMIC-IV**: Physionet MIMIC-IV Demo Dataset
- **PyCaret**: AutoML framework for model training
- **FastAPI**: High-performance Python web framework
- **React**: Frontend JavaScript library
