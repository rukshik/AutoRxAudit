# AutoRxAudit Production Deployment Guide

This guide explains how to deploy the AutoRxAudit system to a production environment.

## Prerequisites

1. Python 3.8 or higher
2. Node.js and npm
3. Access to an Ethereum node (Infura, Alchemy, or your own node)
4. Ethereum account with sufficient funds for deployment
5. Environment variables properly configured

## Environment Setup

1. Set up required environment variables:

```bash
# Ethereum node access
export ETHEREUM_PROVIDER_URL="https://mainnet.infura.io/v3/YOUR-PROJECT-ID"

# Deployment account
export DEPLOYER_PRIVATE_KEY="your-private-key"

# Initial authorized addresses (comma-separated)
export AUTHORIZED_DOCTORS="0xaddress1,0xaddress2"
export AUTHORIZED_PHARMACISTS="0xaddress3,0xaddress4"
export AUTHORIZED_AUDITORS="0xaddress5,0xaddress6"
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

1. Copy the production configuration template:

```bash
cp blockchain_config.production.json blockchain_config.json
```

2. Update the configuration with your specific settings:
   - Choose the appropriate network (mainnet, sepolia, goerli)
   - Set your Infura/Alchemy project ID
   - Configure gas price strategy and confirmation blocks

## Deployment Steps

1. **Deploy the Smart Contract:**

```bash
python deploy_production.py
```

This will:
- Compile the smart contract with optimizations
- Deploy to the specified network
- Configure initial authorized users
- Update the configuration with the deployed contract address

2. **Verify the Deployment:**

- Check deployment.log for any issues
- Verify the contract on Etherscan
- Test basic contract interactions

3. **Configure Application:**

Update your application to use the production configuration:

```python
from blockchain_audit import BlockchainPrescriptionAudit

# Initialize with production config
audit = BlockchainPrescriptionAudit(config_file="blockchain_config.json")
```

## Security Considerations

1. **Private Key Management:**
   - NEVER commit private keys to version control
   - Use a secure key management service in production
   - Rotate keys periodically

2. **Access Control:**
   - Carefully manage authorized addresses
   - Implement proper role separation
   - Monitor contract events for suspicious activity

3. **Network Security:**
   - Use secure connections to Ethereum nodes
   - Implement rate limiting
   - Monitor for failed transactions

4. **Data Privacy:**
   - Ensure patient data is properly anonymized
   - Implement proper access controls
   - Follow healthcare data regulations

## Monitoring and Maintenance

1. **Logging:**
   - Monitor deployment.log
   - Set up log aggregation
   - Configure alerts for critical events

2. **Transaction Monitoring:**
   - Monitor gas costs
   - Track failed transactions
   - Set up alerts for high-risk prescriptions

3. **Regular Maintenance:**
   - Update authorized users as needed
   - Monitor smart contract state
   - Keep dependencies up to date

## Troubleshooting

Common issues and solutions:

1. **Transaction Failures:**
   - Check gas price and limits
   - Verify account balance
   - Check network status

2. **Authorization Issues:**
   - Verify authorized addresses
   - Check transaction sender
   - Confirm role assignments

3. **Network Issues:**
   - Check node connectivity
   - Verify network configuration
   - Monitor node performance

## Support

For production support:
1. Check the logs first
2. Review documentation
3. Contact the development team
4. Open a GitHub issue for non-urgent matters

## Important Notes

- Always test changes on a testnet first
- Keep backup of configuration and deployment data
- Regularly audit system access and permissions
- Monitor gas costs and optimize transactions
- Keep track of contract upgrades and migrations