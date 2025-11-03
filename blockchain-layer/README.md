# AutoRxAudit Blockchain Layer

Immutable audit trail for prescription flagging and pharmacist decisions.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AutoRxAudit System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Frontend (React)  →  Main API (FastAPI)                    │
│       :3000               :8000                              │
│                             ↓                                │
│                    Blockchain Service                        │
│                      (Python/FastAPI)                        │
│                           :8001                              │
│                             ↓                                │
│                      Hardhat Node                            │
│                    (Local Ethereum)                          │
│                          :8545                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Two Separate Components

### 1. **hardhat-node/** - Blockchain Infrastructure
- Local Ethereum blockchain (like PostgreSQL for blockchain)
- Runs at: `http://127.0.0.1:8545`
- Smart contract deployment
- Immutable data storage
- **Technology**: Node.js, Hardhat, Solidity

### 2. **blockchain-service/** - Python Microservice
- REST API for blockchain interaction
- Runs at: `http://localhost:8001`
- Connects main API to blockchain
- Web3.py integration
- **Technology**: Python, FastAPI, Web3.py

## Quick Start (4 Steps)

### Step 1: Install Dependencies

Terminal 1 (Hardhat):
```bash
cd hardhat-node
npm install
```

Terminal 2 (Python Service):
```bash
cd blockchain-service
pip install -r requirements.txt
```

### Step 2: Start Hardhat Node

Keep this terminal running:
```bash
cd hardhat-node
npx hardhat node
```

### Step 3: Deploy Smart Contract

New terminal:
```bash
cd hardhat-node
npx hardhat run scripts/deploy.js --network localhost
```

Copy the `CONTRACT_ADDRESS` from output to `blockchain-service/.env`

### Step 4: Start Blockchain Service

```bash
cd blockchain-service
python blockchain_service.py
```

Test: http://localhost:8001 (should return health check)

## Configuration

### hardhat-node/.env (Optional)
```
DEPLOYER_PRIVATE_KEY=         # Optional for local
SEPOLIA_RPC_URL=              # For production testnet
```

### blockchain-service/.env (Required)
```
BLOCKCHAIN_RPC_URL=http://127.0.0.1:8545
CONTRACT_ADDRESS=0x5FbDB...   # From deployment step
DEPLOYER_PRIVATE_KEY=         # Optional for local
BLOCKCHAIN_SERVICE_PORT=8001
```

## Integration with Main API

Add to `api/app.py`:

```python
import httpx

BLOCKCHAIN_SERVICE_URL = "http://localhost:8001"

async def record_to_blockchain(patient_id: str, drug_name: str, risk_score: float, risk_factors: str):
    """Record flagged prescription to blockchain."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BLOCKCHAIN_SERVICE_URL}/record-flagged-prescription",
            json={
                "patient_id": patient_id,
                "drug_name": drug_name,
                "risk_score": risk_score,
                "risk_factors": risk_factors
            }
        )
        return response.json()

# In audit_prescription endpoint, after flagging:
if flagged:
    blockchain_result = await record_to_blockchain(
        request.patient_id,
        request.drug_name,
        oud_score * 100,  # Convert to 0-100
        flag_reason
    )
    print(f"Recorded to blockchain: TX {blockchain_result['transaction_hash']}")
```

## API Endpoints

### Blockchain Service (Port 8001)

**POST /record-flagged-prescription**
```json
{
  "patient_id": "PAT_00001",
  "drug_name": "Oxycodone 5mg",
  "risk_score": 85.5,
  "risk_factors": "High OUD risk"
}
```

**POST /record-pharmacist-action**
```json
{
  "blockchain_prescription_id": 1,
  "action": "OVERRIDE_APPROVE",
  "action_reason": "Clinical judgment",
  "pharmacist_id": "PHARM_001"
}
```

**GET /prescription/{blockchain_id}**
Returns full prescription record from blockchain.

**GET /prescriptions/all**
Returns all prescriptions (for blockchain viewer page).

## Directory Structure

```
blockchain-layer/
├── hardhat-node/           # Blockchain infrastructure
│   ├── contracts/          # Smart contracts (Solidity)
│   ├── scripts/            # Deployment scripts
│   ├── hardhat.config.js   # Hardhat configuration
│   ├── package.json        # Node.js dependencies
│   ├── .env               # Deployment keys (optional)
│   └── README.md          # Hardhat-specific docs
│
├── blockchain-service/     # Python microservice
│   ├── blockchain_service.py  # FastAPI application
│   ├── requirements.txt   # Python dependencies
│   └── .env              # Configuration (REQUIRED)
│
├── obsolete/             # Old files (ignore)
└── README.md            # This file (instructions for both)
```

## Troubleshooting

### Hardhat node won't start
- Check port 8545: `netstat -an | findstr :8545` (Windows)
- Kill process: `npx hardhat clean`

### Blockchain service connection error
1. Verify Hardhat node is running
2. Check RPC URL in `blockchain-service/.env`: `http://127.0.0.1:8545`
3. Verify CONTRACT_ADDRESS is set

### Transaction fails
- Restart Hardhat node (blockchain state resets)
- Redeploy contract (new address)
- Update CONTRACT_ADDRESS in `blockchain-service/.env`

## Production Deployment

### Option 1: Sepolia Testnet (Free)
1. Get free Sepolia ETH: https://sepoliafaucet.com/
2. Update `hardhat-node/.env`:
   ```
   SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/YOUR_KEY
   SEPOLIA_PRIVATE_KEY=your_private_key
   ```
3. Deploy: `npx hardhat run scripts/deploy.js --network sepolia`
4. Update `blockchain-service/.env` with new CONTRACT_ADDRESS

### Option 2: Azure (Future)
- Deploy blockchain service to Azure App Service
- Use Azure Managed Blockchain or external node provider

## No Subscriptions Needed!

- **Local**: Free (runs on your computer)
- **Sepolia Testnet**: Free (public testnet)
- **Infura**: Free tier (100k requests/day)

## Next Steps

1. ✅ Hardhat and service separated
2. ⏳ Integrate with main API (add HTTP calls)
3. ⏳ Create blockchain viewer page in frontend
4. ⏳ Test end-to-end flow
5. ⏳ Deploy to production (Sepolia testnet)
