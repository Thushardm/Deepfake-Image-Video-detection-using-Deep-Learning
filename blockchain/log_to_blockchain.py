import json
from web3 import Web3
import datetime

def log_detection_to_chain(media_hash, label, confidence):
    with open("blockchain/contract_config.json") as f:
        config = json.load(f)

    web3 = Web3(Web3.HTTPProvider(config["network_rpc"]))
    private_key = config["private_key"]
    sender = config["wallet_address"]

    with open("blockchain/DetectionLoggerABI.json") as f:
        abi = json.load(f)

    contract = web3.eth.contract(address=config["contract_address"], abi=abi)

    # Prepare transaction
    nonce = web3.eth.get_transaction_count(sender)
    txn = contract.functions.logDetection(media_hash, label, int(confidence * 100), int(datetime.datetime.now().timestamp())).build_transaction({
        'chainId': 1337,  # For Ganache
        'gas': 300000,
        'gasPrice': web3.to_wei('20', 'gwei'),
        'nonce': nonce
    })

    signed_txn = web3.eth.account.sign_transaction(txn, private_key=private_key)
    tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)
    return web3.to_hex(tx_hash)
