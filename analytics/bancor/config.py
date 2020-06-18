import os
from multiprocessing.pool import ThreadPool

from pymongo import MongoClient
from web3.auto.ipc import w3

ADDRESSES = {
    'bancor_contract_registry': '0x52Ae12ABe5D8BD778BD5397F99cA900624CfADD4',
    'bnt': '0x1F573D6Fb3F13d689FF844B4cE37794d79a7FF1C',
    'usdb': '0x309627af60F0926daa6041B8279484312f2bf060',
    'eth': '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE'
}

DEPRECATED_TOKENS = {}

REORG_PROTECTION_BLOCKS_COUNT = 50

BNT_DECIMALS = 18

CURRENT_BLOCK = w3.eth.blockNumber - REORG_PROTECTION_BLOCKS_COUNT

LOGS_BLOCKS_CHUNK = 500

HISTORY_CHUNK_SIZE = 5000

DIST_DIR = 'hugo/static/bancor/'

LIQUIDITY_DATA = os.path.join(DIST_DIR, 'data/{}/liquidity.csv')

PROVIDERS_DATA = os.path.join(DIST_DIR, 'data/providers/{}.csv')

PROVIDERS_TOKEN_DATA = os.path.join(DIST_DIR, 'data/providers/{}_token.json')

ROI_DATA = os.path.join(DIST_DIR, 'data/roi/{}.csv')

TOTAL_VOLUME_DATA = os.path.join(DIST_DIR, 'data/{}/total_volume.csv')

TOKENS_DATA = os.path.join(DIST_DIR, 'data/tokens.json')

THREADS = 2

pool = ThreadPool(THREADS)

MONGO_URI = os.environ['MONGO_URI']

MONGO_DATABASE = MONGO_URI.split('/')[-1]

mongo = MongoClient(MONGO_URI, retryWrites=False)

EVENT_PRICE_DATA_UPDATE = '0x8a6a7f53b3c8fa1dc4b83e3f1be668c1b251ff8d44cdcb83eb3acec3fec6a788'

EVENT_CONVERSION = '0x276856b36cbc45526a0ba64f44611557a2a8b68662c5388e9fe6d72e86e1c8cb'

EVENT_VIRTUAL_BALANCE_ENABLED = '0x64622fbd54039f76d87a876ecaea9bdb6b9b493d7a35ca38ae82b53dcddbe2e4'

CONVERTER_EVENTS = [EVENT_PRICE_DATA_UPDATE, EVENT_CONVERSION]

EVENT_TRANSFER = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'

RELAY_EVENTS = [EVENT_TRANSFER]

MODULE_DIR = 'analytics/bancor'

INFOS_DUMP = os.path.join(MODULE_DIR, 'infos.dump')

LAST_BLOCK_DUMP = os.path.join(MODULE_DIR, 'last_block.dump')

TIMESTAMPS_DUMP = os.path.join(MODULE_DIR, 'timestamps.dump')

GRAPHQL_LOGS_QUERY = '''
{{
    logs(filter: {{fromBlock: {fromBlock}, toBlock: {toBlock}, addresses: {addresses}, topics: {topics}}}) {{
    data account {{ address }} topics index transaction {{ block {{ number }} }}
    }}
}}'''

GRAPHQL_ENDPOINT = 'http://localhost:8547/graphql'
