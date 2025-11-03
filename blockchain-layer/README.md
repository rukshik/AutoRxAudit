# Blockchain Layer - AutoRxAudit

Immutable audit trail for prescription flagging and pharmacist actions using Ethereum smart contracts.

## Architecture

```
Hardhat Node (Port 8545)     ← Local Ethereum blockchain
      ↑
      | JSON-RPC
      ↓
Blockchain Service (Port 8001)  ← Python FastAPI microservice
      ↑
      | HTTP
      ↓
Main API (Port 8000)         ← Your main FastAPI application
```

## Quick Start Guide

### Step 1: Install Dependencies

```bash
# Install Node.js dependencies (Hardhat)
cd blockchain-layer
npm install

# Install Python dependencies (Blockchain Service)
pip install -r requirements.txt
```

### Step 2: Start Hardhat Node

Open **Terminal 1**:

```bash
cd blockchain-layer
npx hardhat node
```

**Leave this running!** You should see:
```
Started HTTP and WebSocket JSON-RPC server at http://127.0.0.1:8545/
```

### Step 3: Deploy Smart Contract

Open **Terminal 2**:

```bash
cd blockchain-layer
npx hardhat run scripts/deploy.js --network localhost
```

**Copy the output** and update `.env` file:
```
CONTRACT_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3
```

### Step 4: Start Blockchain Service

Keep Terminal 2 open:

```bash
python blockchain_service.py
```

Service will start on **http://localhost:8001**

### Step 5: Test It

Open browser: http://localhost:8001/docs

Try the endpoints:
- `GET /` - Health check
- `GET /prescriptions/count` - Get total prescriptions
- `POST /record-flagged-prescription` - Record flagged prescription

## Configuration (.env file)

```env
# Blockchain RPC (Hardhat local node)
BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545

# Contract address (from deployment step)
CONTRACT_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3

# Deployer account private key (optional - Hardhat provides accounts)
DEPLOYER_PRIVATE_KEY=

# Blockchain service port
BLOCKCHAIN_SERVICE_PORT=8001
```

## API Endpoints

### Health Check
```
GET /
```

### Record Flagged Prescription
```
POST /record-flagged-prescription
{
  "patient_id": "20000199",
  "drug_name": "Oxycodone 5mg",
  "risk_score": 85,
  "risk_factors": "{\"high_oud_risk\": true}",
  "doctor_id": "DR001",
  "pharmacy_id": "PH001",
  "quantity": 30
}
```

### Record Pharmacist Action
```
POST /record-pharmacist-action
{
  "blockchain_prescription_id": 1,
  "action": "OVERRIDE_APPROVE",
  "action_reason": "Patient has valid chronic pain condition",
  "pharmacist_id": "pharmacist@hospital.com"
}
```

### Get Prescription Record
```
GET /prescription/{blockchain_id}
```

### Get All Prescriptions
```
GET /prescriptions/all?limit=50
```

### Get Prescription Count
```
GET /prescriptions/count
```

## Integration with Main API

Add to your `api/app.py`:

```python
import httpx

BLOCKCHAIN_SERVICE_URL = "http://localhost:8001"

# When prescription is flagged:
async def record_to_blockchain(patient_id, drug_name, risk_score, risk_factors):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BLOCKCHAIN_SERVICE_URL}/record-flagged-prescription",
            json={
                "patient_id": patient_id,
                "drug_name": drug_name,
                "risk_score": int(risk_score * 100),
                "risk_factors": risk_factors
            }
        )
        return response.json()

# When pharmacist takes action:
async def record_pharmacist_action_blockchain(blockchain_id, action, reason, pharmacist_email):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BLOCKCHAIN_SERVICE_URL}/record-pharmacist-action",
            json={
                "blockchain_prescription_id": blockchain_id,
                "action": action,
                "action_reason": reason,
                "pharmacist_id": pharmacist_email
            }
        )
        return response.json()
```

## Production Deployment (Azure)

For production, use **Sepolia testnet** instead of local Hardhat:

1. Get Infura/Alchemy API key (free)
2. Deploy contract to Sepolia testnet
3. Update `.env`:

```env
BLOCKCHAIN_RPC_URL=https://sepolia.infura.io/v3/YOUR_API_KEY
CONTRACT_ADDRESS=0x... (from Sepolia deployment)
DEPLOYER_PRIVATE_KEY=0x... (your private key for Sepolia)
```

## Troubleshooting

### "Cannot connect to blockchain"
- Make sure Hardhat node is running: `npx hardhat node`
- Check RPC URL in `.env` matches Hardhat output

### "CONTRACT_ADDRESS not set"
- Run deployment script: `npx hardhat run scripts/deploy.js --network localhost`
- Copy contract address to `.env`

### "Transaction failed"
- Restart Hardhat node (it resets state)
- Re-deploy contract
- Update CONTRACT_ADDRESS in `.env`

## Development Workflow

1. Start Hardhat: `npx hardhat node` (Terminal 1)
2. Deploy contract: `npx hardhat run scripts/deploy.js --network localhost` (Terminal 2)
3. Start blockchain service: `python blockchain_service.py` (Terminal 2)
4. Integrate with main API
5. Test end-to-end flow

## Files Structure

```
blockchain-layer/
├── .env                          # Configuration (blockchain URL, contract address)
├── hardhat.config.js             # Hardhat configuration
├── package.json                  # Node.js dependencies
├── requirements.txt              # Python dependencies
├── blockchain_service.py         # FastAPI blockchain microservice
├── README.md                     # This file
├── contracts/
│   └── PrescriptionAuditContract.sol  # Smart contract
└── scripts/
    └── deploy.js                 # Deployment script
```

## No Subscriptions Needed!

- ✅ Hardhat: Free, runs locally
- ✅ Sepolia Testnet: Free, public testnet
- ✅ No Azure/AWS blockchain services needed
- ✅ No cryptocurrency needed for local development
- ✅ Testnet ETH is free (from faucets)
