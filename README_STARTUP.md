## Startup Instructions

**Note:** For every terminal, activate the Python virtual environment:

```sh
source .venv/bin/activate
```

---

### 1. Start QKD Service

**Open a new terminal:**
```sh
source .venv/bin/activate
cd qkd-service
python app.py
```

---

### 2. Start Blockchain

#### a. Hardhat Node

**Open a new terminal:**
```sh
source .venv/bin/activate
cd blockchain-layer/hardhat-node
npx hardhat node
```

> **Important:** Ensure one of the private keys from the terminal output is present in the `blockchain-service/.env` file.

**Open another new terminal:**
```sh
source .venv/bin/activate
cd blockchain-layer/hardhat-node
npx hardhat run scripts/deploy-pharmacy-workflow.js --network localhost
```

> **Important:** Make sure the workflow address matches the one in the `blockchain-service/.env` file.

#### b. Blockchain Service

**In the same terminal as above:**
```sh
cd ../blockchain-service
python blockchain_service.py
```

---

### 3. Start Pharmacy App

**Open a new terminal:**
```sh
source .venv/bin/activate
cd pharmacy/pharmacyapp
python app.py
```

---

### 4. Start Doctor App

**Open a new terminal:**
```sh
source .venv/bin/activate
cd doctor/doctorapp
python app.py
```

---

## That's all! The full app is now up and running.