import json
import logging
import os
import pickle
from collections import defaultdict
from itertools import groupby
from math import sqrt
from operator import itemgetter
from typing import List, Dict, Iterable

import requests
from eth_utils import to_checksum_address
from hexbytes import HexBytes
from retrying import retry

from config import w3, LOGS_BLOCKS_CHUNK, CURRENT_BLOCK, pool, CONVERTER_EVENTS, HISTORY_CHUNK_SIZE, \
    EVENT_PRICE_DATA_UPDATE, \
    ADDRESSES, EVENT_CONVERSION, ROI_DATA, BNT_DECIMALS, LIQUIDITY_DATA, TIMESTAMPS_DUMP, TOTAL_VOLUME_DATA, \
    TOKENS_DATA, VOLUME_DATA, RELAY_EVENTS, PROVIDERS_DATA, GRAPHQL_ENDPOINT, GRAPHQL_LOGS_QUERY, INFOS_DUMP, \
    LAST_BLOCK_DUMP, BROKEN_TOKENS
from contracts import BancorConverter, SmartToken, BancorConverterRegistry
from history import History
from relay_info import RelayInfo
from utils import timeit

logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_chart_range(start: int) -> Iterable[int]:
    return range(start, w3.eth.blockNumber, HISTORY_CHUNK_SIZE)


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
                t['details'][0]['decimals'],
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
    converter_addresses = registry.latest_converter_addresses()
    for token_addr, converter_addr in converter_addresses.items():
        converter = BancorConverter(converter_addr)
        relay_token_addr = converter.contract.functions.token().call()
        relay_token = SmartToken(relay_token_addr)
        symbol = relay_token.contract.functions.symbol().call()
        if symbol != 'BNT':
            tokens_data.append(RelayInfo(
                relay_token_addr,
                symbol,
                0,
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
                0,
                data['converterAddress']
            ))

    logging.info('Got info about {} tokens from cotrader {} list'.format(len(tokens_data), list_name))
    return tokens_data


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


# def get_logs(addresses: List[str], topics: List, start_block: int) -> Dict[str, List]:
#     @retry(stop_max_attempt_number=3, wait_fixed=1)
#     def get_chunk(start):
#         return w3.eth.getLogs({
#             'fromBlock': start,
#             'toBlock': min(start + LOGS_BLOCKS_CHUNK - 1, CURRENT_BLOCK),
#             'address': addresses,
#             'topics': topics})
#
#     log_chunks = pool.map(get_chunk, range(start_block, CURRENT_BLOCK + 1, LOGS_BLOCKS_CHUNK))
#     logs = [log for chunk in log_chunks for log in chunk]
#     logs.sort(key=lambda l: (l['address'], l['blockNumber'], l['transactionIndex'], l['logIndex']))
#     return dict((k, list(g)) for k, g in groupby(logs, itemgetter('address')))


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


def invariant(bnt_balance, token_balance, token_supply):
    return sqrt(bnt_balance * token_balance) / token_supply if token_supply else 1


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


@timeit
def populate_history(infos: List[RelayInfo]) -> List[RelayInfo]:
    for info in infos:
        if len(info.converter_logs) == 0:
            logging.warning('No logs for converter {}. Skipping...'.format(info.token_symbol))
            continue
        converter = BancorConverter(info.converter_address)
        info.history = list()
        info.volume = list()
        i = 0
        prev_bnt_balance, prev_token_balance, prev_token_supply = None, None, None
        bnt_balance, token_balance, token_supply = None, None, None
        total_volume_by_trader = defaultdict(int)

        for block_number in get_chart_range(info.converter_logs[0]['blockNumber'] // 5000 * 5000):
            volume_by_trader = defaultdict(int)
            while i < len(info.converter_logs) and info.converter_logs[i]['blockNumber'] <= block_number:
                log = info.converter_logs[i]
                topic = log['topics'][0].hex()
                i += 1
                if topic == EVENT_CONVERSION:
                    event = converter.parse_event('Conversion', log)
                    if event['args']['_fromToken'] == ADDRESSES['bnt']:
                        volume_by_trader[event['args']['_trader']] += event['args']['_amount']
                        total_volume_by_trader[event['args']['_trader']] += event['args']['_amount']
                    else:
                        volume_by_trader[event['args']['_trader']] += event['args']['_return'] + \
                                                                  event['args']['_conversionFee']
                        total_volume_by_trader[event['args']['_trader']] += event['args']['_return'] + \
                                                                        event['args']['_conversionFee']
                elif topic == EVENT_PRICE_DATA_UPDATE:
                    event = converter.parse_event('PriceDataUpdate', log)
                    if event['args']['_connectorToken'] == ADDRESSES['bnt']:
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
                    invariant(bnt_balance, token_balance, token_supply) /
                    invariant(prev_bnt_balance, prev_token_balance, prev_token_supply),
                    bnt_balance,
                    token_balance,
                    sum(volume_by_trader.values()),
                    volume_by_trader
                ))
            else:
                info.history.append(History(block_number, 1, 0, 0, sum(volume_by_trader.values()), volume_by_trader))
            prev_bnt_balance = bnt_balance
            prev_token_balance = token_balance
            prev_token_supply = token_supply

        total_volume = sum(total_volume_by_trader.values())
        valuable_traders = {t for (t, v) in total_volume_by_trader.items() if v > total_volume / 1000}
        info.valuable_traders = list(valuable_traders)
        for history_point in info.history:
            filtered_volume_by_trader = defaultdict(int)
            for (t, v) in history_point.volume_by_trader.items():
                if t in valuable_traders:
                    filtered_volume_by_trader[t] = v
                else:
                    filtered_volume_by_trader['Other'] += v
            history_point.volume_by_trader = filtered_volume_by_trader

        info.bnt_balance = bnt_balance
        info.token_balance = token_balance

    logging.info('Loaded history of {} converters'.format(len(infos)))
    return infos


@timeit
def save_tokens(infos: List[RelayInfo]):
    with open(TOKENS_DATA, 'w') as out_f:
        json.dump({'results': [{'id': info.token_symbol.lower(), 'text': info.token_symbol} for info in infos
                               if info.history]},
                  out_f, indent=1)


@timeit
def save_roi_data(infos: List[RelayInfo], timestamps: Dict[int, int]):
    for info in infos:
        if not info.history:
            continue
        with open(ROI_DATA.format(info.token_symbol.lower()), 'w') as out_f:
            out_f.write('timestamp,ROI,Token Price,Trade Volume\n')
            for history_point in info.history:
                if history_point.bnt_balance == 0:
                    continue
                out_f.write(','.join([str(timestamps[history_point.block_number] * 1000),
                                      '{}'.format(history_point.dm_change),
                                      '{}'.format(history_point.token_balance / history_point.bnt_balance),
                                      '{:.2f}'.format(history_point.trade_volume / 10 ** BNT_DECIMALS)]) + '\n')


@timeit
def save_liquidity_data(infos: List[RelayInfo], timestamps: Dict[int, int]):
    valuable_infos = [info for info in infos if is_valuable(info)]
    other_infos = [info for info in infos if not is_valuable(info)]

    data = defaultdict(dict)

    for info in infos:
        for history_point in info.history:
            data[history_point.block_number][info.token_symbol] = history_point.bnt_balance / 10 ** BNT_DECIMALS

    with open(LIQUIDITY_DATA, 'w') as out_f:
        out_f.write(','.join(['timestamp'] + [i.token_symbol for i in valuable_infos] + ['Other\n']))
        for b, ts in sorted(timestamps.items()):
            out_f.write(','.join([str(ts * 1000)] +
                                 ['{:.2f}'.format(data[b].get(i.token_symbol) or 0) for i in valuable_infos] +
                                 ['{:.2f}'.format(sum(data[b].get(i.token_symbol) or 0 for i in other_infos))]
                                 ) + '\n')


@timeit
def save_volume_data(infos: List[RelayInfo], timestamps: Dict[int, int]):
    for info in infos:
        if not info.history:
            continue
        with open(VOLUME_DATA.format(info.token_symbol.lower()), 'w') as out_f:
            out_f.write(','.join(['timestamp'] + ['\u200b{}'.format(t) for t in info.valuable_traders] +
                                 ['Other']) + '\n')
            for history_point in info.history:
                out_f.write(','.join(
                    [str(timestamps[history_point.block_number] * 1000)] +
                    ['{:.2f}'.format(history_point.volume_by_trader[t] / 10 ** BNT_DECIMALS)
                     if history_point.volume_by_trader[t] else ''
                     for t in info.valuable_traders + ['Other']]
                ) + '\n')


@timeit
def save_total_volume_data(infos: List[RelayInfo], timestamps: Dict[int, int]):
    valuable_infos = infos
    other_infos = []

    data = defaultdict(dict)

    for info in infos:
        for history_point in info.history:
            data[history_point.block_number][info.token_symbol] = history_point.trade_volume / 10 ** BNT_DECIMALS

    with open(TOTAL_VOLUME_DATA, 'w') as out_f:
        out_f.write(','.join(['timestamp'] + [i.token_symbol for i in valuable_infos] + ['Other\n']))
        for b, ts in sorted(timestamps.items()):
            out_f.write(','.join([str(ts * 1000)] +
                                 ['{:.2f}'.format(data[b].get(i.token_symbol) or 0) for i in valuable_infos] +
                                 ['{:.2f}'.format(sum(data[b].get(i.token_symbol) or 0 for i in other_infos))]
                                 ) + '\n')


@timeit
def save_providers_data(infos: List[RelayInfo]):
    for info in infos:
        if not info.history:
            continue
        with open(PROVIDERS_DATA.format(info.token_symbol.lower()), 'w') as out_f:
            out_f.write('provider,bnt\n')
            total_supply = sum(info.providers.values())
            remaining_supply = total_supply
            for p, v in sorted(info.providers.items(), key=lambda x: x[1], reverse=True):
                s = v / total_supply
                if s >= 0.01:
                    out_f.write('\u200b{},{:.2f}\n'.format(p, info.bnt_balance * s / 10 ** BNT_DECIMALS))
                    remaining_supply -= v
            if remaining_supply > 0:
                out_f.write('Other,{:.2f}\n'.format(info.bnt_balance * remaining_supply / total_supply / 10 ** BNT_DECIMALS))


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
    data = get_official_tokens() + get_registry_tokens() + get_cotrader_tokens(True) + get_cotrader_tokens(False)
    for info in data:
        if info.token_symbol in BROKEN_TOKENS:
            continue
        known_info = info_by_token.get(info.token_address)
        if known_info is None:
            new_infos.append(info)
            info_by_token[info.token_address] = info
        else:
            if known_info.converter_address != info.converter_address:
                new_infos.append(info)
                info_by_token[info.token_address] = info
                known_infos.remove(known_info)

    return new_infos


def is_valuable(info: RelayInfo) -> bool:
    return info.bnt_balance >= 10000 * 10 ** BNT_DECIMALS


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
        populate_history(relay_infos)
        populate_providers(relay_infos)
        relay_infos = sorted(relay_infos, key=lambda x: x.bnt_balance, reverse=True)
        pickle_infos(relay_infos)
        pickle_last_block(CURRENT_BLOCK)
    else:
        logging.info('Loaded data is up to date')

    min_block = min(relay_info.history[0].block_number for relay_info in relay_infos if relay_info.history)
    timestamps = unpickle_timestamps()
    timestamps = load_timestamps(min_block, timestamps)
    pickle_timestamps(timestamps)

    valuable_infos = [info for info in relay_infos if is_valuable(info)]

    save_tokens(valuable_infos)
    save_roi_data(valuable_infos, timestamps)
    save_liquidity_data(relay_infos, timestamps)
    save_volume_data(valuable_infos, timestamps)
    save_total_volume_data(valuable_infos, timestamps)
    save_providers_data(valuable_infos)


if __name__ == '__main__':
    main()
