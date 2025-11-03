from web3 import Web3
import json

def deploy_contract():
    # Connect to Ganache
    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:8546'))
    
    if not w3.is_connected():
        raise Exception('Could not connect to Ganache')
    print('Connected to Ganache')
    
    # Set default account (first account is the contract owner)
    w3.eth.default_account = w3.eth.accounts[0]
    
    # First, compile the contract and get ABI/Bytecode
    with open('PrescriptionAuditContract.sol', 'r') as file:
        contract_source = file.read()

    from solcx import compile_source, install_solc
    
    print('Installing solc...')
    install_solc(version='0.8.19')

    print('Compiling contract...')
    compiled_sol = compile_source(
        contract_source,
        output_values=['abi', 'bin'],
        solc_version='0.8.19',
        optimize=True,
        optimize_runs=200,
        via_ir=True
    )

    # Get the contract interface from compilation result
    contract_id, contract_interface = compiled_sol.popitem()
    print('Contract compiled successfully')
    
    # Create the contract object
    contract = w3.eth.contract(
        abi=contract_interface['abi'],
        bytecode=contract_interface['bin']
    )

    # Submit the transaction that deploys the contract
    tx_hash = contract.constructor().transact()
    
    # Wait for the transaction to be mined, and get the transaction receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    print(f'Contract deployed at: {tx_receipt.contractAddress}')
    print(f'Gas used: {tx_receipt.gasUsed}')
    
    return tx_receipt.contractAddress

if __name__ == '__main__':
    contract_address = deploy_contract()