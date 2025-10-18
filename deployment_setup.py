"""
Deployment and Configuration Script for AutoRxAudit Blockchain System
"""

import json
import os
from typing import Dict, Any


class AutoRxAuditConfig:
    """Configuration management for AutoRxAudit system"""

    def __init__(self, config_file: str = "autorxaudit_config.json"):
        self.config_file = config_file
        self.config = self._load_default_config()
        self._load_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "blockchain": {
                "rpc_url": "http://localhost:8545",
                "contract_address": "",
                "private_key": "",
                "gas_limit": 200000,
                "gas_price": 20000000000,  # 20 gwei
                "chain_id": 1337,  # Local development chain
            },
            "ai": {
                "model_path": "best_model_shap_will_get_opioid_rx.pkl",
                "risk_threshold": 70,
                "confidence_threshold": 0.7,
            },
            "pharmacy": {
                "verification_timeout": 30,
                "max_override_attempts": 3,
                "audit_log_retention_days": 365,
            },
            "security": {
                "encryption_enabled": True,
                "audit_log_encryption": True,
                "access_control_enabled": True,
            },
            "logging": {
                "level": "INFO",
                "file": "autorxaudit.log",
                "max_size_mb": 100,
                "backup_count": 5,
            },
        }

    def _load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    file_config = json.load(f)
                    self.config.update(file_config)
                print(f"Configuration loaded from {self.config_file}")
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
        else:
            print(f"Config file {self.config_file} not found, using defaults")
            self._save_config()

    def _save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
            print(f"Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving config file: {e}")

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation"""
        keys = key_path.split(".")
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key_path.split(".")
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        self._save_config()

    def get_blockchain_config(self) -> Dict[str, Any]:
        """Get blockchain configuration"""
        return self.config.get("blockchain", {})

    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI configuration"""
        return self.config.get("ai", {})

    def get_pharmacy_config(self) -> Dict[str, Any]:
        """Get pharmacy configuration"""
        return self.config.get("pharmacy", {})


def create_deployment_script():
    """Create deployment script for AutoRxAudit system"""

    script_content = """#!/bin/bash

# AutoRxAudit Deployment Script
# This script sets up the AutoRxAudit blockchain system

echo "=== AutoRxAudit Deployment Script ==="

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git curl

# Install Node.js and npm for blockchain development
echo "Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Ganache CLI for local blockchain
echo "Installing Ganache CLI..."
sudo npm install -g ganache-cli

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p data/blockchain
mkdir -p models
mkdir -p config

# Initialize configuration
echo "Initializing configuration..."
python3 -c "
from ai_blockchain_integration import AutoRxAuditConfig
config = AutoRxAuditConfig()
print('Configuration initialized')
"

# Start local blockchain
echo "Starting local blockchain..."
ganache-cli --port 8545 --gasLimit 0x1fffffffffffff --gasPrice 0x1 --accounts 10 --defaultBalanceEther 1000 --deterministic > logs/ganache.log 2>&1 &
GANACHE_PID=$!
echo "Ganache started with PID: $GANACHE_PID"
echo $GANACHE_PID > logs/ganache.pid

# Wait for blockchain to start
sleep 5

# Deploy smart contract
echo "Deploying smart contract..."
python3 deploy_contract.py

echo "=== Deployment Complete ==="
echo "Blockchain running on: http://localhost:8545"
echo "Configuration file: autorxaudit_config.json"
echo "Logs directory: logs/"
echo ""
echo "To stop the system:"
echo "  kill \$(cat logs/ganache.pid)"
echo ""
echo "To run the system:"
echo "  source venv/bin/activate"
echo "  python3 ai_blockchain_integration.py"
"""

    with open("deploy.sh", "w") as f:
        f.write(script_content)

    # Make script executable
    os.chmod("deploy.sh", 0o755)
    print("Deployment script created: deploy.sh")


def create_contract_deployment_script():
    """Create smart contract deployment script"""

    script_content = '''"""
Smart Contract Deployment Script for AutoRxAudit
"""

import json
import time
from web3 import Web3
from eth_account import Account
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_contract():
    """Deploy the PrescriptionAuditContract to local blockchain"""
    
    # Connect to local blockchain
    w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
    
    if not w3.is_connected():
        logger.error("Failed to connect to blockchain")
        return None
    
    logger.info("Connected to local blockchain")
    
    # Get first account (Ganache provides 10 accounts)
    accounts = w3.eth.accounts
    deployer_account = accounts[0]
    
    logger.info(f"Deploying from account: {deployer_account}")
    
    # Contract bytecode (this would be compiled from the .sol file)
    # For demo purposes, we'll use a placeholder
    contract_bytecode = "0x608060405234801561001057600080fd5b50600436106100a95760003560e01c8063..."
    
    # Contract ABI (simplified for demo)
    contract_abi = [
        {
            "inputs": [],
            "stateMutability": "nonpayable",
            "type": "constructor"
        },
        {
            "inputs": [
                {"name": "patientId", "type": "string"},
                {"name": "doctorId", "type": "string"},
                {"name": "pharmacyId", "type": "string"},
                {"name": "drugName", "type": "string"},
                {"name": "quantity", "type": "uint256"},
                {"name": "riskScore", "type": "uint8"},
                {"name": "riskFactors", "type": "string"}
            ],
            "name": "recordFlaggedPrescription",
            "outputs": [{"name": "", "type": "uint256"}],
            "stateMutability": "nonpayable",
            "type": "function"
        }
    ]
    
    try:
        # Deploy contract
        contract = w3.eth.contract(bytecode=contract_bytecode, abi=contract_abi)
        
        # Build transaction
        transaction = contract.constructor().build_transaction({
            'from': deployer_account,
            'gas': 2000000,
            'gasPrice': w3.eth.gas_price,
            'nonce': w3.eth.get_transaction_count(deployer_account),
        })
        
        # Sign and send transaction
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key="0x...")  # Ganache default key
        tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
        
        # Wait for deployment
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        contract_address = receipt.contractAddress
        logger.info(f"Contract deployed at: {contract_address}")
        
        # Save contract address to config
        from ai_blockchain_integration import AutoRxAuditConfig
        config = AutoRxAuditConfig()
        config.set("blockchain.contract_address", contract_address)
        
        # Save deployment info
        deployment_info = {
            "contract_address": contract_address,
            "transaction_hash": tx_hash.hex(),
            "block_number": receipt.blockNumber,
            "gas_used": receipt.gasUsed,
            "deployer": deployer_account,
            "deployment_time": int(time.time())
        }
        
        with open("data/blockchain/deployment.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        logger.info("Contract deployment completed successfully")
        return contract_address
        
    except Exception as e:
        logger.error(f"Contract deployment failed: {e}")
        return None

if __name__ == "__main__":
    deploy_contract()
'''

    with open("deploy_contract.py", "w") as f:
        f.write(script_content)

    print("Contract deployment script created: deploy_contract.py")


def create_docker_compose():
    """Create Docker Compose configuration for easy deployment"""

    docker_compose_content = """version: '3.8'

services:
  ganache:
    image: trufflesuite/ganache-cli:latest
    ports:
      - "8545:8545"
    command: ["--port", "8545", "--gasLimit", "0x1fffffffffffff", "--gasPrice", "0x1", "--accounts", "10", "--defaultBalanceEther", "1000", "--deterministic"]
    networks:
      - autorxaudit-network

  autorxaudit-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - BLOCKCHAIN_RPC_URL=http://ganache:8545
      - CONTRACT_ADDRESS=${CONTRACT_ADDRESS}
    depends_on:
      - ganache
    networks:
      - autorxaudit-network
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./models:/app/models

networks:
  autorxaudit-network:
    driver: bridge

volumes:
  autorxaudit-data:
  autorxaudit-logs:
"""

    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose_content)

    print("Docker Compose file created: docker-compose.yml")


def create_dockerfile():
    """Create Dockerfile for containerized deployment"""

    dockerfile_content = """FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data models config

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "ai_blockchain_integration.py"]
"""

    with open("Dockerfile", "w") as f:
        f.write(dockerfile_content)

    print("Dockerfile created: Dockerfile")


def main():
    """Create all deployment files"""
    print("Creating AutoRxAudit deployment files...")

    create_deployment_script()
    create_contract_deployment_script()
    create_docker_compose()
    create_dockerfile()

    print("\n=== Deployment Files Created ===")
    print("1. deploy.sh - Main deployment script")
    print("2. deploy_contract.py - Smart contract deployment")
    print("3. docker-compose.yml - Docker Compose configuration")
    print("4. Dockerfile - Docker container configuration")
    print("\nTo deploy:")
    print("  chmod +x deploy.sh")
    print("  ./deploy.sh")
    print("\nOr with Docker:")
    print("  docker-compose up -d")


if __name__ == "__main__":
    main()
