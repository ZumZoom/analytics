import json
import logging
import os
import pickle
import re
from bisect import bisect
from collections import defaultdict
from itertools import groupby
from functools import reduce
from operator import itemgetter
from typing import List, Dict, Iterable

import requests
from eth_utils import to_checksum_address
from hexbytes import HexBytes
from retrying import retry
from web3.exceptions import BadFunctionCallOutput

from config import w3, LOGS_BLOCKS_CHUNK, CURRENT_BLOCK, pool, CONVERTER_EVENTS, HISTORY_CHUNK_SIZE, \
    EVENT_PRICE_DATA_UPDATE, ADDRESSES, EVENT_CONVERSION, ROI_DATA, BNT_DECIMALS, LIQUIDITY_DATA, TIMESTAMPS_DUMP, \
    TOTAL_VOLUME_DATA, TOKENS_DATA, RELAY_EVENTS, PROVIDERS_DATA, GRAPHQL_ENDPOINT, GRAPHQL_LOGS_QUERY, INFOS_DUMP, \
    LAST_BLOCK_DUMP, DEPRECATED_TOKENS, mongo, MONGO_DATABASE, PROVIDERS_TOKEN_DATA, EVENT_VIRTUAL_BALANCE_ENABLED
from contracts import BancorConverter, SmartToken, BancorConverterRegistry, ERC20
from history import History
from relay_info import RelayInfo
from utils import timeit

logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_base_token(info: RelayInfo) -> str:
    return 'usdb' if re.sub(r'\W|\d|_', '', info.token_symbol).endswith('USDB') else 'bnt'


def get_chart_range(start: int) -> Iterable[int]:
    return range(start, CURRENT_BLOCK, HISTORY_CHUNK_SIZE)


@timeit
def load_timestamps(start: int, stored_timestamps: dict) -> Dict[int, int]:
    for n in get_chart_range(start):
        if n not in stored_timestamps:
            stored_timestamps[n] = w3.eth.getBlock(n)['timestamp']
    return stored_timestamps


@timeit
def get_official_tokens() -> List[RelayInfo]:
    url = 'https://api.bancor.network/0.1/currencies'
    params = {
        'name': 'BNT',
        'limit': 100,
        'excludeEosRelays': 'true',
        'excludeSubTypes': 'bounty',
        'stage': 'traded'
    }
    tokens_data = list()
    for skip_value in range(0, 200, 100):
        params['skip'] = skip_value
        tokens_data.extend([
            RelayInfo(
                to_checksum_address(t['details'][0]['blockchainId']),
                t['details'][0]['symbol'],
                to_checksum_address(t['details'][0]['converter']['blockchainId'])
            )
            for t in requests.get(url, params).json()['data']['currencies']['page']
            if t['code'] != 'BNT'
        ])
    logging.info('Got info about {} official tokens'.format(len(tokens_data)))
    return tokens_data


@timeit
def get_registry_tokens() -> List[RelayInfo]:
    tokens_data = list()
    registry = BancorConverterRegistry(ADDRESSES['bancor_converter_registry'])
    relay_token_addresses = registry.get_liquidity_pools()
    for relay_token_addr in relay_token_addresses:
        relay_token = SmartToken(relay_token_addr)
        symbol = relay_token.contract.functions.symbol().call()
        if symbol != 'BNT':
            tokens_data.append(RelayInfo(
                relay_token_addr,
                symbol,
                relay_token.contract.functions.owner().call()
            ))
    logging.info('Got info about {} tokens from registry'.format(len(tokens_data)))
    return tokens_data


@timeit
def get_cotrader_tokens(official: bool = True) -> List[RelayInfo]:
    list_name = 'official' if official else 'unofficial'
    url = 'https://api-bancor.cotrader.com/' + list_name
    tokens_data = list()
    for data in requests.get(url).json()['result']:
        if data['symbol'] != 'BNT':
            tokens_data.append(RelayInfo(
                data['smartTokenAddress'],
                data['smartTokenSymbol'],
                data['converterAddress']
            ))

    logging.info('Got info about {} tokens from cotrader {} list'.format(len(tokens_data), list_name))
    return tokens_data


@timeit
def populate_token_info(infos: List[RelayInfo]) -> List[RelayInfo]:
    for info in infos:
        if not (hasattr(info, 'token_decimals') and info.token_decimals is not None):
            base_token_address = ADDRESSES[get_base_token(info)]
            try:
                try:
                    token_address = BancorConverter(info.converter_address).connector_tokens(0)
                    if token_address == base_token_address:
                        token_address = BancorConverter(info.converter_address).connector_tokens(1)
                except BadFunctionCallOutput:
                    # Old converter
                    token_address = BancorConverter(info.converter_address).reserve_tokens(0)
                    if token_address == base_token_address:
                        token_address = BancorConverter(info.converter_address).reserve_tokens(1)
            except:
                print('token: {}, converter: {}, relay: {}'.format(
                    info.token_symbol, info.converter_address, info.token_address))
                info.token_decimals = None
                info.underlying_token_symbol = None
                continue
            erc_20 = ERC20(token_address)
            info.token_decimals = erc_20.decimals()
            info.underlying_token_symbol = erc_20.symbol()
            print('Success: token: {}, underlying token: {}, address: {}, decimals: {}'.format(
                info.token_symbol, info.underlying_token_symbol, token_address, info.token_decimals))

    return infos


@timeit
def get_logs(addresses: List[str], topics: List, start_block: int) -> Dict[str, List]:
    @retry(stop_max_attempt_number=3, wait_fixed=1)
    def get_chunk(start):
        resp = requests.post(
            GRAPHQL_ENDPOINT,
            json={'query': GRAPHQL_LOGS_QUERY.format(fromBlock=start,
                                                     toBlock=min(start + LOGS_BLOCKS_CHUNK - 1, CURRENT_BLOCK),
                                                     addresses=json.dumps(addresses),
                                                     topics=json.dumps(topics))}
        )
        return postprocess_graphql_response(resp.json()['data']['logs'])

    log_chunks = pool.map(get_chunk, range(start_block, CURRENT_BLOCK + 1, LOGS_BLOCKS_CHUNK))
    logs = [log for chunk in log_chunks for log in chunk]
    logs.sort(key=lambda l: (l['address'], l['blockNumber'], l['logIndex']))
    return dict((k, list(g)) for k, g in groupby(logs, itemgetter('address')))


def postprocess_graphql_response(logs: List[dict]) -> List[dict]:
    return [{
        'topics': [HexBytes(t) for t in log['topics']],
        'blockNumber': int(log['transaction']['block']['number'], 16),
        'data': log['data'],
        'logIndex': log['index'],
        'transactionIndex': None,
        'transactionHash': None,
        'address': to_checksum_address(log['account']['address']),
        'blockHash': None
    } for log in logs]


@timeit
def load_logs(start_block: int, infos: List[RelayInfo]) -> List[RelayInfo]:
    converter_addresses = [info.converter_address for info in infos]
    converter_logs = get_logs(converter_addresses, [CONVERTER_EVENTS], start_block)
    relay_addresses = [info.token_address for info in infos]
    relay_logs = get_logs(relay_addresses, [RELAY_EVENTS], start_block)

    for info in infos:
        new_converter_logs = converter_logs.get(info.converter_address)
        if new_converter_logs:
            info.converter_logs += new_converter_logs
        new_relay_logs = relay_logs.get(info.token_address)
        if new_relay_logs:
            info.relay_logs += new_relay_logs

    logging.info('Loaded logs for {} converters'.format(len(infos)))
    return infos


def invariant(balances, token_supply):
    return pow(reduce(lambda x, y: x * y, balances), 1 / len(balances)) / token_supply if token_supply else 1


@timeit
def fix_vb_issues(infos: List[RelayInfo]) -> List[RelayInfo]:
    vb_logs = w3.eth.getLogs({
        'fromBlock': 0,
        'toBlock': 'latest',
        'topics': [EVENT_VIRTUAL_BALANCE_ENABLED]
    })
    vb_logs = dict((k, list(g)) for k, g in groupby(vb_logs, itemgetter('address')))
    for info in infos:
        vb_logs_i = vb_logs.get(info.converter_address)
        if vb_logs_i:
            converter = BancorConverter(info.converter_address)
            for log in vb_logs_i[::-1]:
                event = converter.parse_event('VirtualBalancesEnable', log)
                if not event['args']['_enabled']:
                    # Remove all logs before this one
                    all_blocks = [log['blockNumber'] for log in info.converter_logs]
                    pos = bisect(all_blocks, event['blockNumber'])
                    info.converter_logs = info.converter_logs[pos:]
                    logging.info('Removed {} logs before {} for {} {}'.format(
                        len(all_blocks) - len(info.converter_logs),
                        event['blockNumber'],
                        info.token_symbol,
                        info.converter_address
                    ))
                    break
    return infos


@timeit
def populate_providers(infos: List[RelayInfo]) -> List[RelayInfo]:
    for info in infos:
        token = SmartToken(info.token_address)
        info.providers = defaultdict(int)
        for log in info.relay_logs:
            event = token.parse_event('Transfer', log)
            if event['args']['_from'] == info.token_address:
                info.providers[event['args']['_to']] += event['args']['_value']
            elif event['args']['_to'] == info.token_address:
                info.providers[event['args']['_from']] -= event['args']['_value']
            else:
                info.providers[event['args']['_from']] -= event['args']['_value']
                info.providers[event['args']['_to']] += event['args']['_value']
    logging.info('Loaded info about providers of {} exchanges'.format(len(infos)))
    return infos


def multireserve_populate_history(info: RelayInfo):
    converter = BancorConverter(info.converter_address)
    token_count = converter.connector_token_count()
    init_block = None
    info.history = list()
    i = 0

    prev_balances, prev_token_supply = dict(), None
    balances, token_supply = dict(), None

    for block_number in get_chart_range(
            info.converter_logs[0]['blockNumber'] // HISTORY_CHUNK_SIZE * HISTORY_CHUNK_SIZE):
        volume = 0
        while i < len(info.converter_logs) and info.converter_logs[i]['blockNumber'] <= block_number:
            log = info.converter_logs[i]
            topic = log['topics'][0].hex()
            i += 1
            if topic == EVENT_CONVERSION:
                event = converter.parse_event('Conversion', log)
                if event['args']['_fromToken'] in {'0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
                                                   '0xdAC17F958D2ee523a2206206994597C13D831ec7'}:
                    volume += event['args']['_amount'] * 1000000000000
            elif topic == EVENT_PRICE_DATA_UPDATE:
                event = converter.parse_event('PriceDataUpdate', log)
                connector_token = event['args']['_connectorToken']
                connector_balance = event['args']['_connectorBalance']
                balances[connector_token] = connector_balance
                token_supply = event['args']['_tokenSupply']
                if not init_block or init_block == info.converter_logs[i-1]['blockNumber']:
                    prev_balances[connector_token] = connector_balance
                    if not init_block and len(prev_balances) == token_count:
                        init_block = info.converter_logs[i-1]['blockNumber']
                    prev_token_supply = token_supply

        if len(prev_balances) == token_count:
            info.history.append(History(
                block_number,
                invariant(balances.values(), token_supply) /
                invariant(prev_balances.values(), prev_token_supply),
                1,
                1,
                volume
            ))
        else:
            info.history.append(History(block_number, 1, 0, 0, 0))
        prev_balances = balances.copy()
        prev_token_supply = token_supply

    info.bnt_balance = 0
    info.token_balance = 0


@timeit
def populate_history(infos: List[RelayInfo]) -> List[RelayInfo]:
    for info in infos:
        if info.token_symbol in {'USDARY', 'USDRAY1'}:
            multireserve_populate_history(info)
            continue

        if len(info.converter_logs) == 0:
            logging.warning('No logs for converter {}. Skipping...'.format(info.token_symbol))
            continue
        base_token = get_base_token(info)
        converter = BancorConverter(info.converter_address)
        info.history = list()
        info.volume = list()
        i = 0
        prev_bnt_balance, prev_token_balance, prev_token_supply = None, None, None
        bnt_balance, token_balance, token_supply = None, None, None

        for block_number in get_chart_range(
                info.converter_logs[0]['blockNumber'] // HISTORY_CHUNK_SIZE * HISTORY_CHUNK_SIZE):
            volume = 0
            while i < len(info.converter_logs) and info.converter_logs[i]['blockNumber'] <= block_number:
                log = info.converter_logs[i]
                topic = log['topics'][0].hex()
                i += 1
                if topic == EVENT_CONVERSION:
                    event = converter.parse_event('Conversion', log)
                    if event['args']['_fromToken'] == ADDRESSES[base_token]:
                        volume += event['args']['_amount']
                    else:
                        volume += event['args']['_return'] + event['args']['_conversionFee']
                elif topic == EVENT_PRICE_DATA_UPDATE:
                    event = converter.parse_event('PriceDataUpdate', log)
                    if event['args']['_connectorToken'] == ADDRESSES[base_token]:
                        bnt_balance = event['args']['_connectorBalance']
                        if prev_bnt_balance is None:
                            prev_bnt_balance = bnt_balance
                    else:
                        token_balance = event['args']['_connectorBalance']
                        if prev_token_balance is None:
                            prev_token_balance = token_balance
                    token_supply = event['args']['_tokenSupply']
                    if prev_token_supply is None:
                        prev_token_supply = token_supply

            if prev_bnt_balance is not None and prev_token_balance is not None:
                info.history.append(History(
                    block_number,
                    invariant([bnt_balance, token_balance], token_supply) /
                    invariant([prev_bnt_balance, prev_token_balance], prev_token_supply),
                    bnt_balance,
                    token_balance,
                    volume
                ))
            else:
                info.history.append(History(block_number, 1, 0, 0, volume))
            prev_bnt_balance = bnt_balance
            prev_token_balance = token_balance
            prev_token_supply = token_supply

        info.bnt_balance = bnt_balance or 0
        info.token_balance = token_balance or 0

    logging.info('Loaded history of {} converters'.format(len(infos)))
    return infos


@timeit
def save_tokens(infos: List[RelayInfo], path: str):
    results = list()
    counts = defaultdict(int)
    for info in infos:
        if info.history:
            ticker_name = re.sub('[\s/]', '', info.token_symbol.lower())
            count = counts[ticker_name]
            counts[ticker_name] += 1
            if count > 0:
                ticker_name = ticker_name + '_' + str(count)
            results.append({'id': ticker_name, 'text': info.token_symbol})
    with open(path, 'w') as out_f:
        json.dump({'results': results}, out_f)


@timeit
def save_roi_data(infos: List[RelayInfo], timestamps: Dict[int, int]):
    counts = defaultdict(int)
    for info in infos:
        if not info.history:
            continue
        ticker_name = re.sub('[\s/]', '', info.token_symbol.lower())
        count = counts[ticker_name]
        counts[ticker_name] += 1
        if count > 0:
            ticker_name = ticker_name + '_' + str(count)
        with open(ROI_DATA.format(ticker_name), 'w') as out_f:
            out_f.write('timestamp,ROI,Token Price,Trade Volume\n')
            for history_point in info.history:
                if history_point.bnt_balance == 0:
                    continue
                out_f.write(','.join([str(timestamps[history_point.block_number] * 1000),
                                      '{}'.format(history_point.dm_change),
                                      '{}'.format(history_point.token_balance / history_point.bnt_balance),
                                      '{:.2f}'.format(history_point.trade_volume / 10 ** BNT_DECIMALS)]) + '\n')


@timeit
def save_liquidity_data(infos: List[RelayInfo], timestamps: Dict[int, int], base_token: str):
    valuable_infos = [info for info in infos if is_valuable(info, base_token)]
    other_infos = [info for info in infos if not is_valuable(info, base_token)]

    data = defaultdict(dict)

    for info in infos:
        for history_point in info.history:
            data[history_point.block_number][info.converter_address] = history_point.bnt_balance / 10 ** BNT_DECIMALS

    with open(LIQUIDITY_DATA.format(base_token, base_token), 'w') as out_f:
        out_f.write(','.join(['timestamp'] + [i.token_symbol for i in valuable_infos] + ['Other\n']))
        first_row = True
        for b, ts in sorted(timestamps.items()):
            if first_row and sum(data[b].get(i.converter_address) or 0 for i in other_infos) + \
                    sum(data[b].get(i.converter_address) or 0 for i in valuable_infos) == 0:
                continue
            first_row = False
            out_f.write(','.join([str(ts * 1000)] +
                                 ['{:.2f}'.format(data[b].get(i.converter_address) or 0) for i in valuable_infos] +
                                 ['{:.2f}'.format(sum(data[b].get(i.converter_address) or 0 for i in other_infos))]
                                 ) + '\n')


@timeit
def save_total_volume_data(infos: List[RelayInfo], timestamps: Dict[int, int], base_token: str):
    valuable_infos = [info for info in infos if is_valuable(info, base_token)]
    other_infos = [info for info in infos if not is_valuable(info, base_token)]

    data = defaultdict(dict)

    for info in infos:
        for history_point in info.history:
            data[history_point.block_number][info.converter_address] = history_point.trade_volume / 10 ** BNT_DECIMALS

    with open(TOTAL_VOLUME_DATA.format(base_token), 'w') as out_f:
        out_f.write(','.join(['timestamp'] + [i.token_symbol for i in valuable_infos] + ['Other\n']))
        first_row = True
        for b, ts in sorted(timestamps.items()):
            if first_row and sum(data[b].get(i.converter_address) or 0 for i in other_infos) + \
                    sum(data[b].get(i.converter_address) or 0 for i in valuable_infos) == 0:
                continue
            first_row = False
            out_f.write(','.join([str(ts * 1000)] +
                                 ['{:.2f}'.format(data[b].get(i.converter_address) or 0) for i in valuable_infos] +
                                 ['{:.2f}'.format(sum(data[b].get(i.converter_address) or 0 for i in other_infos))]
                                 ) + '\n')


@timeit
def save_providers_data(infos: List[RelayInfo]):
    counts = defaultdict(int)
    for info in infos:
        if not info.history:
            continue
        ticker_name = re.sub('[\s/]', '', info.token_symbol.lower())
        count = counts[ticker_name]
        counts[ticker_name] += 1
        if count > 0:
            ticker_name = ticker_name + '_' + str(count)
        with open(PROVIDERS_TOKEN_DATA.format(ticker_name), 'w') as out_f:
            data = {
                'token_name': info.underlying_token_symbol,
                'total_tokens': info.token_balance / 10 ** info.token_decimals
            }
            json.dump(data, out_f)
        with open(PROVIDERS_DATA.format(ticker_name), 'w') as out_f:
            out_f.write('provider,bnt\n')
            total_supply = sum(info.providers.values())
            remaining_supply = total_supply
            if total_supply == 0:
                for p, v in sorted(info.providers.items(), key=lambda x: x[1], reverse=True):
                    out_f.write('\u200b{},{:.2f}\n'.format(p, info.bnt_balance * s / 10 ** BNT_DECIMALS))
                continue
            for p, v in sorted(info.providers.items(), key=lambda x: x[1], reverse=True):
                s = v / total_supply
                if s >= 0.01:
                    out_f.write('\u200b{},{:.2f}\n'.format(p, info.bnt_balance * s / 10 ** BNT_DECIMALS))
                    remaining_supply -= v
            if remaining_supply > 0:
                out_f.write(
                    'Other,{:.2f}\n'.format(info.bnt_balance * remaining_supply / total_supply / 10 ** BNT_DECIMALS))


def pickle_timestamps(timestamps: Dict[int, int]):
    with open(TIMESTAMPS_DUMP, 'wb') as out_f:
        pickle.dump(timestamps, out_f)


def unpickle_timestamps() -> Dict[int, int]:
    if os.path.exists(TIMESTAMPS_DUMP):
        with open(TIMESTAMPS_DUMP, 'rb') as in_f:
            return pickle.load(in_f)
    else:
        return {}


def pickle_infos(infos: List[RelayInfo]):
    with open(INFOS_DUMP, 'wb') as out_f:
        pickle.dump(infos, out_f)


def unpickle_infos() -> List[RelayInfo]:
    if os.path.exists(INFOS_DUMP):
        with open(INFOS_DUMP, 'rb') as in_f:
            return pickle.load(in_f)
    else:
        return []


def pickle_last_block(block_number: int):
    with open(LAST_BLOCK_DUMP, 'wb') as out_f:
        pickle.dump(block_number, out_f)


def unpickle_last_block() -> int:
    if os.path.exists(LAST_BLOCK_DUMP):
        with open(LAST_BLOCK_DUMP, 'rb') as in_f:
            return pickle.load(in_f)
    else:
        return 0


def update_required(last_processed_block: int) -> bool:
    return CURRENT_BLOCK // HISTORY_CHUNK_SIZE * HISTORY_CHUNK_SIZE > last_processed_block


def load_new_infos(known_infos: List[RelayInfo]) -> List[RelayInfo]:
    info_by_token = {info.token_address: info for info in known_infos}
    new_infos = []
    data = get_registry_tokens()  # + get_official_tokens() + get_cotrader_tokens(True) + get_cotrader_tokens(False)
    for info in data:
        known_info = info_by_token.get(info.token_address)
        if info.token_symbol in DEPRECATED_TOKENS:
            if known_info is not None:
                known_infos.remove(known_info)
                info_by_token.pop(info.token_address)
            continue
        if known_info is None:
            new_infos.append(info)
            info_by_token[info.token_address] = info
        else:
            if known_info.converter_address != info.converter_address:
                # read converter address from blockchain
                info.converter_address = SmartToken(info.token_address).contract.functions.owner().call()
                try:
                    known_infos.remove(known_info)
                    new_infos.append(info)
                    info_by_token[info.token_address] = info
                except ValueError:
                    # info's converter address was already updated
                    pass

    # delete deprecated data
    info_by_token = {info.token_address: info for info in data}
    for info in known_infos.copy():
        if info_by_token.get(info.token_address) is None:
            known_infos.remove(info)

    return new_infos


def is_valuable(info: RelayInfo, base_token: str) -> bool:
    return info.bnt_balance >= 20000 * 10 ** BNT_DECIMALS if base_token == 'bnt' else not is_empty(info)


def is_empty(info: RelayInfo) -> bool:
    return info.bnt_balance == 0


def save_tokens_to_mongo(infos: List[RelayInfo]):
    tokens_collection = mongo[MONGO_DATABASE].tokens
    tokens_collection.drop()
    tokens = [
        {
            'token': info.token_symbol.lower(),
            'converter': info.converter_address,
            'base_token': get_base_token(info)
        }
        for info in infos
    ]
    tokens_collection.insert_many(tokens)


def save_providers_to_mongo(infos: List[RelayInfo]):
    providers_collection = mongo[MONGO_DATABASE].providers
    providers_collection.drop()
    entries = []

    for info in infos:
        if not info.history:
            continue

        total_supply = sum(info.providers.values())

        if total_supply == 0:
            continue
        remaining_supply = total_supply

        base_token = get_base_token(info)

        for p, v in sorted(info.providers.items(), key=lambda x: x[1], reverse=True):
            s = v / total_supply
            if s >= 0.01:
                entries.append({
                    'token': info.token_symbol.lower(),
                    'base_token': base_token,
                    'provider': p,
                    'base_token_balance': info.bnt_balance * s / 10 ** BNT_DECIMALS,
                })
                remaining_supply -= v
        if remaining_supply > 0:
            entries.append({
                'token': info.token_symbol.lower(),
                'base_token': base_token,
                'provider': 'Other',
                'base_token_balance': info.bnt_balance * remaining_supply / total_supply / 10 ** BNT_DECIMALS,
            })
    providers_collection.insert_many(entries)


def save_history_to_mongo(infos: List[RelayInfo], timestamps: Dict[int, int]):
    history_collection = mongo[MONGO_DATABASE].history
    history_collection.drop()
    entries = []

    for info in infos:
        if not info.history:
            continue

        base_token = get_base_token(info)

        for history_point in info.history:
            if history_point.bnt_balance == 0:
                continue

            entries.append({
                'token': info.token_symbol.lower(),
                'base_token': base_token,
                'timestamp': timestamps[history_point.block_number],
                'gm_change': history_point.dm_change,
                'price': history_point.token_balance / history_point.bnt_balance,
                'volume': history_point.trade_volume / 10 ** BNT_DECIMALS,
                'base_token_balance': history_point.bnt_balance / 10 ** BNT_DECIMALS,
            })
    history_collection.insert_many(entries)


def main():
    saved_block = unpickle_last_block()
    relay_infos = unpickle_infos()
    if update_required(saved_block):
        logging.info('Last seen block: {}, current block: {}, loading data for {} blocks...'.format(
            saved_block, CURRENT_BLOCK, CURRENT_BLOCK - saved_block))
        new_infos = load_new_infos(relay_infos)
        logging.info('Updating {} seen tokens and {} new tokens'.format(len(relay_infos), len(new_infos)))
        if new_infos:
            load_logs(0, new_infos)
        if relay_infos:
            load_logs(saved_block + 1, relay_infos)
        relay_infos += new_infos
        # fix_vb_issues(relay_infos)
        populate_token_info(relay_infos)
        populate_history(relay_infos)
        populate_providers(relay_infos)
        relay_infos = sorted(relay_infos, key=lambda x: x.bnt_balance or 0, reverse=True)
        pickle_infos(relay_infos)
        pickle_last_block(CURRENT_BLOCK)
    else:
        logging.info('Loaded data is up to date')

    min_block = min(relay_info.history[0].block_number for relay_info in relay_infos if relay_info.history)
    timestamps = unpickle_timestamps()
    timestamps = load_timestamps(min_block, timestamps)
    pickle_timestamps(timestamps)

    for base_token in ['bnt', 'usdb']:
        infos = list(filter(lambda info: get_base_token(info) == base_token, relay_infos))
        save_liquidity_data(infos, timestamps, base_token)
        save_total_volume_data(infos, timestamps, base_token)

    not_empty_infos = [info for info in relay_infos if not is_empty(info)]
    multireserve_infos = [info for info in relay_infos if info.token_symbol in {'USDARY', 'USDRAY1'}]
    save_tokens(not_empty_infos + multireserve_infos, TOKENS_DATA)
    save_roi_data(not_empty_infos + multireserve_infos, timestamps)
    save_providers_data(not_empty_infos)

    save_tokens_to_mongo(not_empty_infos)
    save_history_to_mongo(not_empty_infos, timestamps)
    save_providers_to_mongo(not_empty_infos)


if __name__ == '__main__':
    main()
