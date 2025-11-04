# QKD Service - Quantum Key Distribution

## Overview

This service implements the **BB84 Quantum Key Distribution protocol** using Qiskit to provide quantum-safe key exchange 

## What is QKD?

Quantum Key Distribution (QKD) uses quantum mechanics to securely distribute encryption keys between two parties. Any attempt to eavesdrop on the quantum channel will disturb the quantum states, making the eavesdropping detectable.

## BB84 Protocol

The BB84 protocol, invented by Bennett and Brassard in 1984, works as follows:

1. **Alice (sender)** prepares qubits in random states using random bases
2. **Bob (receiver)** measures the qubits using randomly chosen bases
3. **Basis reconciliation**: Alice and Bob publicly compare which bases they used
4. **Sifting**: They keep only the bits where their bases matched
5. **Error checking**: They compare a sample of bits to detect eavesdropping
6. **Privacy amplification**: They hash the remaining bits to generate the final key

## Installation

```bash
cd qkd-service
pip install -r requirements.txt
```

## Configuration

Edit `.env` file:

```
QKD_SERVICE_PORT=8005
QKD_INITIAL_QUBITS=512
QKD_TARGET_KEY_BITS=256
QKD_ERROR_THRESHOLD=0.11
QKD_SESSION_TIMEOUT_SECONDS=300
QKD_ENABLE_EAVESDROP_SIMULATION=false
```

## Running the Service

```bash
python app.py
```

Service will run on `http://localhost:8005`

## API Endpoints

### 1. Initiate QKD Session

```http
POST /qkd/initiate
Content-Type: application/json

{
  "sender": "doctor-office-1",
  "receiver": "pharmacy-A",
  "prescription_uuid": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response:**
```json
{
  "session_id": "qkd-a1b2c3d4e5f6",
  "status": "initiated",
  "message": "QKD session initiated"
}
```

### 2. Complete Key Exchange

```http
POST /qkd/exchange
Content-Type: application/json

{
  "session_id": "qkd-a1b2c3d4e5f6"
}
```

**Response:**
```json
{
  "session_id": "qkd-a1b2c3d4e5f6",
  "success": true,
  "error_rate": 0.02,
  "secure": true,
  "sifted_bits": 256,
  "final_key_bits": 256,
  "message": "Key exchange successful!"
}
```

### 3. Get Encryption Key

```http
POST /qkd/get-key
Content-Type: application/json

{
  "session_id": "qkd-a1b2c3d4e5f6",
  "party": "sender"
}
```

**Response:**
```json
{
  "session_id": "qkd-a1b2c3d4e5f6",
  "key": "a4f3e8d2c1b9...",
  "key_id": "key-abc123",
  "expires_at": "2025-11-03T11:00:00"
}
```

### 4. Get Session Status

```http
GET /qkd/session/{session_id}
```

### 5. Destroy Session

```http
DELETE /qkd/destroy/{session_id}
```

### 6. Get Statistics

```http
GET /qkd/statistics
```

## Usage Flow

```python
# 1. Doctor initiates QKD session
response = requests.post('http://localhost:8005/qkd/initiate', json={
    'sender': 'doctor-office-1',
    'receiver': 'pharmacy-A',
    'prescription_uuid': '550e8400-...'
})
session_id = response.json()['session_id']

# 2. Complete BB84 key exchange
response = requests.post('http://localhost:8005/qkd/exchange', json={
    'session_id': session_id
})

# 3. Doctor gets encryption key
response = requests.post('http://localhost:8005/qkd/get-key', json={
    'session_id': session_id,
    'party': 'sender'
})
doctor_key = response.json()['key']

# 4. Pharmacy gets decryption key (same key!)
response = requests.post('http://localhost:8005/qkd/get-key', json={
    'session_id': session_id,
    'party': 'receiver'
})
pharmacy_key = response.json()['key']

# Keys are identical! Use for AES encryption/decryption

# 5. After use, destroy keys
requests.delete(f'http://localhost:8005/qkd/destroy/{session_id}')
```

## Security Features

- **Quantum-safe**: Key distribution protected by quantum mechanics
- **Eavesdropping detection**: 25% error rate if Eve intercepts
- **One-time keys**: Fresh key for each prescription
- **Perfect forward secrecy**: Keys immediately destroyed after use
- **No key storage**: Keys exist only transiently during exchange

## Testing BB84 Protocol

```bash
# Test the BB84 implementation directly
python bb84.py
```

This will run two tests:
1. Normal key exchange (should succeed)
2. Key exchange with simulated eavesdropper (should detect and fail)

## How It Works

### Without Eavesdropper:
```
Alice sends qubits → Bob measures → Bases match ~50% → Error rate ~0-2% → ✓ Secure
```

### With Eavesdropper (Eve):
```
Alice sends qubits → Eve intercepts → Eve measures (disturbs states) 
→ Eve resends → Bob measures → Error rate ~25% → ⚠️ Detected!
```

## Architecture

```
Doctor App (Alice)          QKD Service          Pharmacy App (Bob)
       |                         |                         |
       |--- Initiate QKD ------->|                         |
       |<--- Session ID ----------|                         |
       |                         |                         |
       |--- Request Exchange --->|<--- Request Exchange ---|
       |                  [Run BB84 Protocol]              |
       |<--- Key (sender) -------|                         |
       |                         |---- Key (receiver) ---->|
       |                         |                         |
    [Encrypt]                                         [Decrypt]
       |                         |                         |
       |--- Destroy Keys ------->|<--- Destroy Keys -------|
```

## Qiskit Implementation

The BB84 protocol uses Qiskit quantum circuits:

- **Rectilinear basis (+)**: States |0⟩ and |1⟩
- **Diagonal basis (x)**: States |+⟩ and |-⟩ (Hadamard-transformed)
- **AerSimulator**: Simulates quantum measurements
- **Quantum circuits**: One per qubit, executed independently

## Performance

- **Initial qubits**: 512
- **Sifting efficiency**: ~50% (basis matching)
- **After error checking**: ~200-250 bits
- **Final key**: 256 bits (SHA-256 hashed)
- **Processing time**: ~1-2 seconds per key exchange

## Production Considerations

For production deployment:
- Replace in-memory storage with Redis
- Add authentication/authorization
- Use real quantum hardware (ID Quantique, Toshiba)
- Implement key pools for high-volume scenarios
- Add rate limiting
- Monitor error rates for security alerts
