# Hardhat Node

Local Ethereum development blockchain for AutoRxAudit.

## Purpose

This is the blockchain infrastructure layer - a local Ethereum node that runs at `http://127.0.0.1:8545`. Think of it like PostgreSQL for blockchain - it's the database that stores the immutable audit trail.

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Hardhat Node

In a dedicated terminal (keep it running):

```bash
npx hardhat node
```

You'll see:
- Started HTTP and WebSocket JSON-RPC server at http://127.0.0.1:8545/
- 20 test accounts with 10000 ETH each

### 3. Deploy Smart Contract

In a new terminal:

```bash
npx hardhat run scripts/deploy.js --network localhost
```

Output will show:
```
Deploying PrescriptionAuditContract...
✓ Contract deployed to: 0x5FbDB2315678afecb367f032d93F642f64180aa3

Add this to blockchain-service/.env:
CONTRACT_ADDRESS=0x5FbDB2315678afecb367f032d93F642f64180aa3
DEPLOYER_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
```

**Important**: Copy the `CONTRACT_ADDRESS` to `blockchain-service/.env`

## File Structure

```
hardhat-node/
├── hardhat.config.js    # Hardhat configuration
├── package.json         # Node.js dependencies
├── contracts/           # Solidity smart contracts
│   └── PrescriptionAuditContract.sol
├── scripts/             # Deployment scripts
│   └── deploy.js
├── .env                 # Optional: deployment keys
└── README.md           # This file
```

## Configuration

### Local Development (Default)

Hardhat automatically creates:
- Network: localhost
- RPC URL: http://127.0.0.1:8545
- Chain ID: 1337
- 20 pre-funded accounts (10000 ETH each)

No configuration needed!

### Production (Sepolia Testnet)

1. Get free Sepolia ETH from faucet: https://sepoliafaucet.com/
2. Create `.env` file:
   ```
   SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/YOUR_INFURA_KEY
   SEPOLIA_PRIVATE_KEY=your_private_key_here
   ```
3. Deploy:
   ```bash
   npx hardhat run scripts/deploy.js --network sepolia
   ```

## Smart Contract

**PrescriptionAuditContract.sol** - Immutable prescription audit trail

Key Functions:
- `recordFlaggedPrescription()` - Record flagged prescription to blockchain
- `overridePrescription()` - Record pharmacist override decision
- `prescriptions(id)` - Query prescription record by ID

Events:
- `PrescriptionFlagged` - Emitted when prescription is flagged
- `PrescriptionOverridden` - Emitted when pharmacist overrides

## Troubleshooting

### Port 8545 already in use
```bash
# Kill existing Hardhat process
npx hardhat clean
```

### Reset blockchain state
```bash
# Stop node (Ctrl+C)
# Restart node
npx hardhat node
```

### Can't connect to node
1. Check node is running: `netstat -an | findstr :8545` (Windows)
2. Verify URL: http://127.0.0.1:8545 (not localhost)
3. Check firewall settings

## No Subscriptions Needed!

- **Local Development**: Free (runs on your computer)
- **Production (Sepolia)**: Free testnet (no subscriptions)

## Next Steps

After deploying the contract:
1. Copy `CONTRACT_ADDRESS` to `blockchain-service/.env`
2. Start blockchain service: `cd ../blockchain-service && python blockchain_service.py`
3. Integrate with main API
