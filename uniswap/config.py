import json
import os
from multiprocessing.pool import ThreadPool
from web3.auto.ipc import w3 as web3

ETH = 10 ** 18

UNISWAP_BEGIN_BLOCK = 6627917

HISTORY_BEGIN_BLOCK = 6628000

HISTORY_CHUNK_SIZE = 5000

REORG_PROTECTION_BLOCKS_COUNT = 50

CURRENT_BLOCK = web3.eth.blockNumber - REORG_PROTECTION_BLOCKS_COUNT

LOGS_BLOCKS_CHUNK = 500

THREADS = 2

pool = ThreadPool(THREADS)

with open('abi/uniswap_factory.abi') as in_f:
    UNISWAP_FACTORY_ABI = json.load(in_f)

with open('abi/uniswap_exchange.abi') as in_f:
    UNISWAP_EXCHANGE_ABI = json.load(in_f)

with open('abi/erc_20.abi') as in_f:
    ERC_20_ABI = json.load(in_f)

with open('abi/str_erc_20.abi') as in_f:
    STR_ERC_20_ABI = json.load(in_f)

with open('abi/str_caps_erc_20.abi') as in_f:
    STR_CAPS_ERC_20_ABI = json.load(in_f)

UNISWAP_FACTORY_ADDRESS = '0xc0a47dFe034B400B47bDaD5FecDa2621de6c4d95'

uniswap_factory = web3.eth.contract(abi=UNISWAP_FACTORY_ABI, address=UNISWAP_FACTORY_ADDRESS)

HARDCODED_INFO = {
    '0xE0B7927c4aF23765Cb51314A0E0521A9645F0E2A': ('DGD', 'DGD', 9),
    '0xBB9bc244D798123fDe783fCc1C72d3Bb8C189413': ('TheDAO', 'TheDAO', 16),
    '0x42456D7084eacF4083f1140d3229471bbA2949A8': ('Synth sETH', 'sETH old', 18),
    '0x89d24A6b4CcB1B6fAA2625fE562bDD9a23260359': ('Sai Stablecoin v1.0', 'SAI', 18),
    '0xF5DCe57282A584D2746FaF1593d3121Fcac444dC': ('Compound Sai', 'cSAI', 8),
    '0x09617F6fD6cF8A71278ec86e23bBab29C04353a7': ('Unblocked Ledger Token', 'ULT (ST)', 18),
    '0x8dd5fbCe2F6a956C3022bA3663759011Dd51e73E': ('TrueUSD', 'TUSD old', 18),
    '0x0cBE2dF57CA9191B64a7Af3baa3F946fa7Df2F25': ('Synth sUSD', 'sUSD old', 18),
}

DIST_DIR = '../hugo/static/uniswap/'

LIQUIDITY_DATA = os.path.join(DIST_DIR, 'data/liquidity.csv')

PROVIDERS_DATA = os.path.join(DIST_DIR, 'data/providers/{}.csv')

ROI_DATA = os.path.join(DIST_DIR, 'data/roi/{}.csv')

VOLUME_DATA = os.path.join(DIST_DIR, 'data/volume/{}.csv')

TOTAL_VOLUME_DATA = os.path.join(DIST_DIR, 'data/total_volume.csv')

TOKENS_DATA = os.path.join(DIST_DIR, 'data/tokens.json')

EVENT_TRANSFER = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'

EVENT_TOKEN_PURCHASE = '0xcd60aa75dea3072fbc07ae6d7d856b5dc5f4eee88854f5b4abf7b680ef8bc50f'

EVENT_ETH_PURCHASE = '0x7f4091b46c33e918a0f3aa42307641d17bb67029427a5369e54b353984238705'

EVENT_ADD_LIQUIDITY = '0x06239653922ac7bea6aa2b19dc486b9361821d37712eb796adfd38d81de278ca'

EVENT_REMOVE_LIQUIDITY = '0x0fbf06c058b90cb038a618f8c2acbf6145f8b3570fd1fa56abb8f0f3f05b36e8'

ALL_EVENTS = [EVENT_TRANSFER, EVENT_TOKEN_PURCHASE, EVENT_ETH_PURCHASE, EVENT_ADD_LIQUIDITY, EVENT_REMOVE_LIQUIDITY]

INFOS_DUMP = 'infos.dump'

LAST_BLOCK_DUMP = 'last_block.dump'

GRAPHQL_LOGS_QUERY = '''
{{
    logs(filter: {{fromBlock: {fromBlock}, toBlock: {toBlock}, addresses: {addresses}, topics: {topics}}}) {{
    data account {{ address }} topics index transaction {{ block {{ number }} }}
    }}
}}'''

GRAPHQL_ENDPOINT = 'http://localhost:8547/graphql'
