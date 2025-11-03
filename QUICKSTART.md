# AutoRxAudit - Ready to Run! 🚀

## ✅ What's Been Completed

### Backend API (Completed in Previous Session)
- ✅ FastAPI application with dual AI models
- ✅ Eligibility Model: 81.94% AUC
- ✅ OUD Risk Model: 99.87% AUC
- ✅ Feature engineering (43 features from MIMIC-IV data)
- ✅ Business rules for prescription flagging
- ✅ PostgreSQL database integration

### Frontend (Just Completed!)
- ✅ React app with 3 main components
- ✅ Login page with authentication
- ✅ Prescription audit form with dropdowns
- ✅ Results display with AI scores
- ✅ Action recording (Approve/Deny/Override)
- ✅ Audit history table

### Database Schema (Ready to Apply)
- ✅ `users` table schema created
- ✅ `audit_actions` table schema created
- ✅ 3 demo users with credentials
- ✅ SQL file ready: `api/database/schema_with_users.sql`

---

## 🚀 Quick Start (3 Steps)

### Step 1: Apply Database Schema
```bash
cd api/database
psql -h autorxaudit-server.postgres.database.azure.com -U cloudsa -d mimiciv_demo_raw -f schema_with_users.sql
```

### Step 2: Start Backend
```bash
cd api
python -m uvicorn app:app --reload
```

### Step 3: Start Frontend
```bash
cd frontend
npm start
```

---

## 🔑 Demo Login Credentials

```
Email: doctor@hospital.com
Password: password123
```

---

## 📋 What You'll See

### 1. Login Page
- Purple gradient background
- White login card
- Email/password fields
- Demo credentials displayed

### 2. Prescription Form
- Patient ID dropdown (from database)
- Drug dropdown (opioids + non-opioids)
- "Audit Prescription" button
- Real-time AI analysis

### 3. Results Display
- **Eligibility Score**: Clinical need indicator (0-100%)
- **OUD Risk Score**: Addiction risk indicator (0-100%)
- **Flag Status**: 🚩 FLAGGED or ✅ APPROVED
- **Recommendation**: Plain English explanation
- **Action Buttons**: Approve/Deny/Override

### 4. Audit History
- Table of all past audits
- Columns: Date, Patient, Drug, Scores, Action, Clinician, Reason
- Color-coded actions
- Sortable data

---

## 🎯 Test Cases to Try

### Case 1: Appropriate Opioid (Should Approve)
- Patient: 20038695 (has pain diagnosis)
- Drug: Oxycodone
- Expected: ✅ APPROVED
- Action: Click "Approve Prescription"

### Case 2: High Risk Patient (Should Flag)
- Select patient with substance abuse history
- Drug: Oxycodone
- Expected: 🚩 FLAGGED
- Action: Review and decide

### Case 3: Non-Opioid Medication
- Any patient
- Drug: Ibuprofen
- Expected: Varies by patient
- Action: Review and approve/deny

---

## 📁 Project Structure

```
AutoRxAudit/
├── api/
│   ├── app.py                    # FastAPI backend (750+ lines)
│   ├── database/
│   │   └── schema_with_users.sql # User authentication schema
│   └── models/
│       ├── eligibility_model.pkl
│       └── oud_model.pkl
│
├── frontend/
│   └── src/
│       ├── App.js                # Main routing
│       ├── Login.js              # Authentication
│       ├── PrescriptionForm.js   # Audit form (230 lines)
│       └── AuditHistory.js       # History table (140 lines)
│
├── SETUP_GUIDE.md               # Comprehensive guide
└── README.md                     # Project overview
```

---

## 🔗 API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/login` | User authentication |
| GET | `/api/patients` | Patient dropdown data |
| GET | `/api/drugs` | Drug dropdown data |
| POST | `/audit-prescription` | AI model inference |
| POST | `/api/audit-action` | Record user decision |
| GET | `/api/audit-history` | Fetch audit records |

---

## 🎨 Features Highlights

### Authentication
- Simple email/password login
- Session persistence in localStorage
- Logout functionality
- User info displayed in header

### Prescription Audit
- Patient selection from database
- Mixed drug list (opioids + non-opioids)
- Dual AI model analysis
- Real-time results with explanations
- Score visualization with color coding

### Action Recording
- 4 action types: Approve, Deny, Override Approve, Override Deny
- Optional reason text field
- Timestamp tracking
- User attribution

### Audit History
- Comprehensive audit log
- Searchable/filterable table
- Color-coded actions
- Responsive design

---

## 🔧 Technology Stack

**Frontend:**
- React 18
- CSS Grid/Flexbox
- Fetch API
- localStorage

**Backend:**
- FastAPI
- PyCaret (AI models)
- PostgreSQL
- Pandas/NumPy

**Database:**
- PostgreSQL on Azure
- MIMIC-IV Demo dataset
- Custom audit tables

---

## ⚠️ Important Notes

### Security (Development Only)
- ⚠️ Plain text passwords
- ⚠️ No JWT tokens
- ⚠️ localStorage sessions
- ⚠️ No HTTPS

**DO NOT USE IN PRODUCTION WITHOUT:**
1. Password hashing (bcrypt)
2. JWT authentication
3. HTTPS/SSL
4. Secure session management
5. CORS restrictions

### Database Schema
**Must run `schema_with_users.sql` before first use!**
This creates the required `users` and `audit_actions` tables.

---

## 🐛 Troubleshooting

### "Cannot connect to server"
- Ensure backend is running on http://localhost:8000
- Check terminal for backend errors

### "Invalid credentials"
- Verify database schema was applied
- Check user exists in `users` table
- Confirm password is "password123"

### "Patient not found"
- Verify MIMIC-IV data is loaded
- Check patient exists in database

### React app won't start
```bash
cd frontend
rm -rf node_modules
npm install
npm start
```

---

## 📈 Next Steps

1. **Test the application** with all demo scenarios
2. **Apply database schema** if not already done
3. **Review audit history** to see recorded actions
4. **Explore enhancements** (see SETUP_GUIDE.md)

---

## 📚 Documentation Files

- **SETUP_GUIDE.md** - Comprehensive setup instructions
- **frontend/README.md** - Frontend-specific documentation
- **api/README.md** - Backend API documentation (if exists)

---

## 🎉 You're All Set!

The AutoRxAudit application is complete and ready to run!

Just execute the 3 quick start steps above and you'll have:
- ✅ AI-powered prescription auditing
- ✅ User authentication
- ✅ Real-time risk assessment
- ✅ Action recording and audit trails

**Enjoy using AutoRxAudit!** 🏥💊🤖
