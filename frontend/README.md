# AutoRxAudit Frontend

React frontend for the AutoRxAudit prescription auditing system with AI-powered risk assessment.

## Features

- **Simple Login**: Email/password authentication
- **Prescription Audit Form**: Patient ID & drug dropdowns with AI analysis
- **Dual Model Inference**: Eligibility (81.94% AUC) + OUD Risk (99.87% AUC)
- **Action Recording**: Approve/Deny/Override with optional reason
- **Audit History**: View all past prescription audits with timestamps

## Quick Start

### 1. Apply Database Schema (First Time Only)

```bash
cd api/database
psql -h autorxaudit-server.postgres.database.azure.com -U cloudsa -d mimiciv_demo_raw -f schema_with_users.sql
```

### 2. Start Backend API

```bash
cd api
python -m uvicorn app:app --reload
```

Backend runs on `http://localhost:8000`

### 3. Start Frontend

```bash
cd frontend
npm install  # First time only
npm start
```

Frontend opens at `http://localhost:3000`

## Demo Accounts

| Email | Password | Role |
|-------|----------|------|
| doctor@hospital.com | password123 | Doctor |
| pharmacist@hospital.com | password123 | Pharmacist |
| admin@hospital.com | admin123 | Admin |

## Usage Flow

1. **Login** with demo account
2. **Select Patient** from dropdown
3. **Select Drug** (opioids or non-opioids)
4. **Audit Prescription** - Get AI analysis
5. **Review Results**: Eligibility score, OUD risk score, flag status
6. **Take Action**: Approve/Deny/Override with optional reason
7. **View History**: See all past audits

## API Endpoints

- `POST /api/login` - Authentication
- `GET /api/patients` - Patient list
- `GET /api/drugs` - Drug list
- `POST /audit-prescription` - AI inference
- `POST /api/audit-action` - Record decision
- `GET /api/audit-history` - Audit records

## Available Scripts

### `npm start`

Runs the app in development mode at [http://localhost:3000](http://localhost:3000).

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
