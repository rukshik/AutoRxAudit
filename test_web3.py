from web3 import Web3

# Connect to Ganache
w3 = Web3(Web3.HTTPProvider('http://localhost:8546'))

# Verify connection
print(f"Connected to Ganache: {w3.is_connected()}")
print(f"ChainId: {w3.eth.chain_id}")

# Get accounts
accounts = w3.eth.accounts
print("\nAccounts:")
for i, account in enumerate(accounts):
    balance = w3.eth.get_balance(account)
    print(f"{i}: {account} - Balance: {w3.from_wei(balance, 'ether')} ETH")