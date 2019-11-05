import os
from multiprocessing.pool import ThreadPool

from pymongo import MongoClient
from web3.auto.ipc import w3

ADDRESSES = {
    'bancor_converter_registry': '0x0DDFF327ddF7fE838e3e63d02001ef23ad1EdE8e',
    'bnt': '0x1F573D6Fb3F13d689FF844B4cE37794d79a7FF1C',
    'usdb': '0x309627af60F0926daa6041B8279484312f2bf060',
}

DEPRECATED_TOKENS = {
    'BMCBNT', 'TIXBNT', 'NPXSBNT', 'DBETBNT', 'NEXOBNT', 'TIOBNT', 'CHXBNT', 'ULTBNT', 'JOYBNT', 'SWMBNT', 'GOLDBNT',
    'COTUSDB2'
}

REORG_PROTECTION_BLOCKS_COUNT = 50

BNT_DECIMALS = 18

CURRENT_BLOCK = w3.eth.blockNumber - REORG_PROTECTION_BLOCKS_COUNT

LOGS_BLOCKS_CHUNK = 2000

HISTORY_CHUNK_SIZE = 5000

DIST_DIR = '../dist/bancor/'

LIQUIDITY_DATA = os.path.join(DIST_DIR, 'data/{}/liquidity.csv')

PROVIDERS_DATA = os.path.join(DIST_DIR, 'data/{}/providers/{}.csv')

ROI_DATA = os.path.join(DIST_DIR, 'data/{}/roi/{}.csv')

TOTAL_VOLUME_DATA = os.path.join(DIST_DIR, 'data/{}/total_volume.csv')

TOKENS_DATA = os.path.join(DIST_DIR, 'data/{}/tokens.json')

THREADS = 2

pool = ThreadPool(THREADS)

MONGO_URI = os.environ['MONGO_URI']

MONGO_DATABASE = MONGO_URI.split('/')[-1]

mongo = MongoClient(MONGO_URI, retryWrites=False)

EVENT_PRICE_DATA_UPDATE = '0x8a6a7f53b3c8fa1dc4b83e3f1be668c1b251ff8d44cdcb83eb3acec3fec6a788'

EVENT_CONVERSION = '0x276856b36cbc45526a0ba64f44611557a2a8b68662c5388e9fe6d72e86e1c8cb'

CONVERTER_EVENTS = [EVENT_PRICE_DATA_UPDATE, EVENT_CONVERSION]

EVENT_TRANSFER = '0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef'

RELAY_EVENTS = [EVENT_TRANSFER]

INFOS_DUMP = 'infos.dump'

LAST_BLOCK_DUMP = 'last_block.dump'

TIMESTAMPS_DUMP = 'timestamps.dump'

GRAPHQL_LOGS_QUERY = '''
{{
    logs(filter: {{fromBlock: {fromBlock}, toBlock: {toBlock}, addresses: {addresses}, topics: {topics}}}) {{
    data account {{ address }} topics index transaction {{ block {{ number }} }}
    }}
}}'''

GRAPHQL_ENDPOINT = 'http://localhost:8547/graphql'
