import json
import logging
import os
import pickle
import re
from collections import defaultdict
from itertools import groupby
from math import sqrt
from operator import itemgetter
from typing import List, Iterable, Dict

import requests
from retrying import retry
from web3._utils.events import get_event_data
from web3.main import HexBytes, to_checksum_address

from analytics.uniswap.config import uniswap_factory, web3, pool, UNISWAP_EXCHANGE_ABI, STR_ERC_20_ABI, ETH, \
    HARDCODED_INFO, STR_CAPS_ERC_20_ABI, ERC_20_ABI, HISTORY_BEGIN_BLOCK, CURRENT_BLOCK, HISTORY_CHUNK_SIZE, \
    LIQUIDITY_DATA, PROVIDERS_DATA, TOKENS_DATA, INFOS_DUMP, LAST_BLOCK_DUMP, ALL_EVENTS, EVENT_TRANSFER, \
    EVENT_ADD_LIQUIDITY, EVENT_REMOVE_LIQUIDITY, EVENT_ETH_PURCHASE, ROI_DATA, EVENT_TOKEN_PURCHASE, VOLUME_DATA, \
    TOTAL_VOLUME_DATA, GRAPHQL_ENDPOINT, GRAPHQL_LOGS_QUERY, LOGS_BLOCKS_CHUNK
from analytics.uniswap.exchange_info import ExchangeInfo
from analytics.uniswap.roi_info import RoiInfo
from analytics.utils import timeit, bytes_to_str


@timeit
def load_token_count() -> int:
    return uniswap_factory.functions.tokenCount().call()


@timeit
def load_tokens(token_count: int) -> List[str]:
    if not token_count:
        token_count = load_token_count()
    tokens = [uniswap_factory.functions.getTokenWithId(i).call() for i in range(1, token_count + 1)]
    logging.info('Found {} tokens'.format(len(tokens)))
    return tokens


@timeit
def load_exchanges(tokens: List[str]) -> List[str]:
    if not tokens:
        tokens = load_tokens()
    exchanges = [uniswap_factory.functions.getExchange(t).call() for t in tokens]
    logging.info('Found {} exchanges'.format(len(exchanges)))
    return exchanges


def load_exchange_data_impl(token_address, exchange_address):
    token = web3.eth.contract(abi=STR_ERC_20_ABI, address=token_address)
    if token_address in HARDCODED_INFO:
        token_name, token_symbol, token_decimals = HARDCODED_INFO[token_address]
    else:
        try:
            token_name = token.functions.name().call()
            token_symbol = token.functions.symbol().call()
            token_decimals = token.functions.decimals().call()
        except Exception:
            try:
                token = web3.eth.contract(abi=STR_CAPS_ERC_20_ABI, address=token_address)
                token_name = token.functions.NAME().call()
                token_symbol = token.functions.SYMBOL().call()
                token_decimals = token.functions.DECIMALS().call()
            except Exception:
                try:
                    token = web3.eth.contract(abi=ERC_20_ABI, address=token_address)
                    token_name = bytes_to_str(token.functions.name().call())
                    token_symbol = bytes_to_str(token.functions.symbol().call())
                    token_decimals = token.functions.decimals().call()
                except Exception:
                    logging.warning('FUCKED UP {}'.format(token_address))
                    return None

    try:
        token_balance = token.functions.balanceOf(exchange_address).call(block_identifier=CURRENT_BLOCK)
    except Exception:
        logging.warning('FUCKED UP {}'.format(token_address))
        return None
    token_symbol = token_symbol.strip('\x00')
    eth_balance = web3.eth.getBalance(exchange_address, block_identifier=CURRENT_BLOCK)
    return ExchangeInfo(token_address,
                        token_name,
                        token_symbol,
                        token_decimals,
                        exchange_address,
                        eth_balance,
                        token_balance)


@timeit
def load_exchange_infos(infos: List[ExchangeInfo]) -> List[ExchangeInfo]:
    token_count = load_token_count()
    tokens = load_tokens(token_count)
    exchanges = load_exchanges(tokens)

    new_infos = filter(None, [load_exchange_data_impl(t, e) for (t, e) in zip(tokens, exchanges)])
    if infos:
        known_tokens = dict((info.token_address, info) for info in infos)
        for new_info in new_infos:
            info = known_tokens.get(new_info.token_address)
            if info:
                info.token_name = new_info.token_name
                info.token_symbol = new_info.token_symbol
                info.token_decimals = new_info.token_decimals
                info.eth_balance = new_info.eth_balance
                info.token_balance = new_info.token_balance
            else:
                infos.append(new_info)
    else:
        infos += new_infos

    logging.info('Loaded info about {} exchanges'.format(len(exchanges)))
    return infos


def get_chart_range(start: int = HISTORY_BEGIN_BLOCK) -> Iterable[int]:
    return range(start, CURRENT_BLOCK, HISTORY_CHUNK_SIZE)


@timeit
def load_timestamps() -> List[int]:
    return [web3.eth.getBlock(n)['timestamp'] for n in get_chart_range()]


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

    _addresses = addresses
    len_addresses = len(addresses)
    len_part = len_addresses // 10
    log_chunks = []
    for i in range(0, len_addresses, len_part):
        addresses = _addresses[i:i + len_part]
        log_chunks += pool.map(get_chunk, range(start_block, CURRENT_BLOCK + 1, LOGS_BLOCKS_CHUNK))
    logs = [log for chunk in log_chunks for log in chunk]
    logs.sort(key=lambda l: (l['address'], l['blockNumber']))
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
def load_logs(start_block: int, infos: List[ExchangeInfo]) -> List[ExchangeInfo]:
    exchange_addresses = [info.exchange_address for info in infos]
    token_addresses = [info.token_address for info in infos]
    exchange_logs = get_logs(exchange_addresses, [ALL_EVENTS], start_block)
    exchange_addresses_topics = ['0x000000000000000000000000' + addr[2:] for addr in exchange_addresses]
    token_logs = get_logs(token_addresses, [[EVENT_TRANSFER], [], exchange_addresses_topics], start_block)
    for info in infos:
        new_exchange_logs = exchange_logs.get(info.exchange_address)
        if new_exchange_logs:
            info.logs += new_exchange_logs
        new_token_logs = token_logs.get(info.token_address)
        if new_token_logs:
            info.logs.extend(filter(transfers_to_address_only(info.exchange_address), new_token_logs))
        info.logs.sort(key=lambda l: (l['blockNumber'], l['logIndex']))

    logging.info('Loaded transfer logs for {} exchanges'.format(len(infos)))
    return infos


def transfers_to_address_only(address: str):
    def foo(log):
        topic_to = log['topics'][2].hex()
        return address == to_checksum_address(topic_to[:2] + topic_to[26:])

    return foo


@timeit
def populate_providers(infos: List[ExchangeInfo]) -> List[ExchangeInfo]:
    for info in infos:
        exchange = web3.eth.contract(abi=UNISWAP_EXCHANGE_ABI, address=info.exchange_address)
        info.providers = defaultdict(int)
        for log in info.logs:
            if log['topics'][0].hex() != EVENT_TRANSFER or log['address'] != info.exchange_address:
                continue
            event = get_event_data(web3.codec, exchange.events.Transfer._get_event_abi(), log)
            if event['args']['from'] == '0x0000000000000000000000000000000000000000':
                info.providers[event['args']['to']] += event['args']['value']
            elif event['args']['to'] == '0x0000000000000000000000000000000000000000':
                info.providers[event['args']['from']] -= event['args']['value']
            else:
                info.providers[event['args']['from']] -= event['args']['value']
                info.providers[event['args']['to']] += event['args']['value']
    logging.info('Loaded info about providers of {} exchanges'.format(len(infos)))
    return infos


def skip_transfer(info: ExchangeInfo, log: dict, i: int) -> bool:
    if i == len(info.logs):
        # last log is legit
        return False

    next_log = info.logs[i]
    if next_log['blockNumber'] > log['blockNumber']:
        # if next log from another block this log is legit
        return False

    if next_log['topics'][0].hex() in {EVENT_ETH_PURCHASE, EVENT_ADD_LIQUIDITY}:
        # if next log is ETH_PURCHASE or ADD_LIQUIDITY then this log is its part
        return True

    if i + 1 == len(info.logs):
        # if only two logs left, it is legit
        return False

    next_next_log = info.logs[i + 1]
    if next_next_log['blockNumber'] > log['blockNumber']:
        # if next next log is from another block then this log is legit
        return False

    if next_log['topics'][0].hex() == EVENT_TRANSFER and next_log['address'] == info.exchange_address and \
            next_next_log['topics'][0].hex() == EVENT_ETH_PURCHASE:
        return True

    return False


@timeit
def populate_roi(infos: List[ExchangeInfo]) -> List[ExchangeInfo]:
    for info in infos:
        try:
            info.roi = list()
            info.history = list()
            exchange = web3.eth.contract(abi=UNISWAP_EXCHANGE_ABI, address=info.exchange_address)
            i = 0
            eth_balance, token_balance = 0, 0
            for block_number in get_chart_range():
                dm_numerator, dm_denominator, trade_volume = 1, 1, 0
                while i < len(info.logs) and info.logs[i]['blockNumber'] <= block_number:
                    log = info.logs[i]
                    topic = log['topics'][0].hex()
                    i += 1
                    if topic == EVENT_TRANSFER:
                        if log['address'] == info.exchange_address:
                            # skip liquidity token transfers
                            continue
                        elif skip_transfer(info, log, i):
                            continue
                        else:
                            event = get_event_data(web3.codec, exchange.events.Transfer._get_event_abi(), log)
                            if event['args']['to'] != info.exchange_address:
                                continue
                            if token_balance > 0:
                                dm_numerator *= token_balance + event['args']['value']
                                dm_denominator *= token_balance
                            token_balance += event['args']['value']
                    elif topic == EVENT_ADD_LIQUIDITY:
                        event = get_event_data(web3.codec, exchange.events.AddLiquidity._get_event_abi(), log)
                        eth_balance += event['args']['eth_amount']
                        token_balance += event['args']['token_amount']
                    elif topic == EVENT_REMOVE_LIQUIDITY:
                        event = get_event_data(web3.codec, exchange.events.RemoveLiquidity._get_event_abi(), log)
                        eth_balance -= event['args']['eth_amount']
                        token_balance -= event['args']['token_amount']
                    elif topic == EVENT_ETH_PURCHASE:
                        event = get_event_data(web3.codec, exchange.events.EthPurchase._get_event_abi(), log)
                        eth_new_balance = eth_balance - event['args']['eth_bought']
                        token_new_balance = token_balance + event['args']['tokens_sold']
                        dm_numerator *= eth_new_balance * token_new_balance
                        dm_denominator *= eth_balance * token_balance
                        trade_volume += event['args']['eth_bought'] / 0.997
                        eth_balance = eth_new_balance
                        token_balance = token_new_balance
                    else:
                        event = get_event_data(web3.codec, exchange.events.TokenPurchase._get_event_abi(), log)
                        eth_new_balance = eth_balance + event['args']['eth_sold']
                        token_new_balance = token_balance - event['args']['tokens_bought']
                        dm_numerator *= eth_new_balance * token_new_balance
                        dm_denominator *= eth_balance * token_balance
                        trade_volume += event['args']['eth_sold']
                        eth_balance = eth_new_balance
                        token_balance = token_new_balance

                info.roi.append(RoiInfo(sqrt(dm_numerator / dm_denominator), eth_balance, token_balance, trade_volume))
                info.history.append(eth_balance)
        except Exception:
            logging.warning('FUCKED UP {} {}'.format(info.token_symbol, info.token_address))

    logging.info('Loaded info about roi of {} exchanges'.format(len(infos)))
    return infos


@timeit
def populate_volume(infos: List[ExchangeInfo]) -> List[ExchangeInfo]:
    for info in infos:
        volume = list()
        info.volume = list()
        exchange = web3.eth.contract(abi=UNISWAP_EXCHANGE_ABI, address=info.exchange_address)
        i = 0
        total_trade_volume = defaultdict(int)
        for block_number in get_chart_range():
            trade_volume = defaultdict(int)
            while i < len(info.logs) and info.logs[i]['blockNumber'] < block_number:
                log = info.logs[i]
                i += 1
                topic = log['topics'][0].hex()
                if topic == EVENT_ETH_PURCHASE:
                    event = get_event_data(web3.codec, exchange.events.EthPurchase._get_event_abi(), log)
                    trade_volume[event['args']['buyer']] += event['args']['eth_bought'] / 0.997
                    total_trade_volume[event['args']['buyer']] += event['args']['eth_bought'] / 0.997
                elif topic == EVENT_TOKEN_PURCHASE:
                    event = get_event_data(web3.codec, exchange.events.TokenPurchase._get_event_abi(), log)
                    trade_volume[event['args']['buyer']] += event['args']['eth_sold']
                    total_trade_volume[event['args']['buyer']] += event['args']['eth_sold']

            volume.append(trade_volume)

        total_volume = sum(total_trade_volume.values())
        valuable_traders = {t for (t, v) in total_trade_volume.items() if v > total_volume / 1000}
        info.valuable_traders = list(valuable_traders)
        for vol in volume:
            filtered_vol = defaultdict(int)
            for (t, v) in vol.items():
                if t in valuable_traders:
                    filtered_vol[t] = v
                else:
                    filtered_vol['Other'] += v
            info.volume.append(filtered_vol)

    logging.info('Volumes of {} exchanges populated'.format(len(infos)))
    return infos


def is_valuable(info: ExchangeInfo) -> bool:
    return info.eth_balance >= 200 * ETH


def is_empty(info: ExchangeInfo) -> bool:
    return info.eth_balance <= ETH


def save_tokens(infos: List[ExchangeInfo], path: str):
    with open(path, 'w') as out_f:
        json.dump({
            'results': [
                {
                    'id': re.sub('[\\s/]', '', info.token_symbol.lower()),
                    'text': info.token_symbol
                }
                for info in infos
            ]
        }, out_f)


def save_liquidity_data(infos: List[ExchangeInfo], timestamps: List[int]):
    if not timestamps:
        timestamps = load_timestamps()

    valuable_infos = [info for info in infos if is_valuable(info)]
    other_infos = [info for info in infos if not is_valuable(info)]

    with open(LIQUIDITY_DATA, 'w') as out_f:
        out_f.write(','.join(['timestamp'] + [i.token_symbol for i in valuable_infos] + ['Other\n']))
        for j in range(len(timestamps)):
            out_f.write(','.join([str(timestamps[j] * 1000)] +
                                 ['{:.2f}'.format(i.history[j]) for i in valuable_infos] +
                                 ['{:.2f}'.format(sum(i.history[j] for i in other_infos))]
                                 ) + '\n')


def save_providers_data(infos: List[ExchangeInfo]):
    for info in infos:
        ticker_name = re.sub('[\\s/]', '', info.token_symbol.lower())
        with open(PROVIDERS_DATA.format(ticker_name), 'w') as out_f:
            out_f.write('provider,eth\n')
            total_supply = sum(info.providers.values())
            remaining_supply = total_supply
            for p, v in sorted(info.providers.items(), key=lambda x: x[1], reverse=True):
                s = v / total_supply
                if s >= 0.01:
                    out_f.write('\u200b{},{:.2f}\n'.format(p, info.eth_balance * s / ETH))
                    remaining_supply -= v
            if remaining_supply > 0:
                out_f.write('Other,{:.2f}\n'.format(info.eth_balance * remaining_supply / total_supply / ETH))


def save_roi_data(infos: List[ExchangeInfo], timestamps: List[int]):
    if not timestamps:
        timestamps = load_timestamps()

    for info in infos:
        ticker_name = re.sub('[\\s/]', '', info.token_symbol.lower())
        with open(ROI_DATA.format(ticker_name), 'w') as out_f:
            out_f.write('timestamp,ROI,Token Price,Trade Volume\n')
            for j in range(len(timestamps)):
                if info.roi[j].eth_balance == 0:
                    continue
                out_f.write(','.join([str(timestamps[j] * 1000),
                                      '{}'.format(info.roi[j].dm_change),
                                      '{}'.format(info.roi[j].token_balance / info.roi[j].eth_balance),
                                      '{:.2f}'.format(info.roi[j].trade_volume / ETH)]) + '\n')


def save_volume_data(infos: List[ExchangeInfo], timestamps: List[int]):
    if not timestamps:
        timestamps = load_timestamps()

    for info in infos:
        ticker_name = re.sub('[\\s/]', '', info.token_symbol.lower())
        with open(VOLUME_DATA.format(ticker_name), 'w') as out_f:
            out_f.write(','.join(['timestamp'] + ['\u200b{}'.format(t) for t in info.valuable_traders] +
                                 ['Other']) + '\n')
            for j in range(len(timestamps)):
                if sum(info.volume[j].values()) == 0:
                    continue
                out_f.write(','.join([str(timestamps[j] * 1000)] +
                                     ['{:.2f}'.format(info.volume[j][t] / ETH) if info.volume[j][t] else ''
                                      for t in info.valuable_traders + ['Other']]) + '\n')


def save_total_volume_data(infos: List[ExchangeInfo], timestamps: List[int]):
    if not timestamps:
        timestamps = load_timestamps()

    valuable_infos = [info for info in infos if is_valuable(info)]
    other_infos = [info for info in infos if not is_valuable(info)]

    with open(TOTAL_VOLUME_DATA, 'w') as out_f:
        out_f.write(','.join(['timestamp'] + [i.token_symbol for i in valuable_infos] + ['Other\n']))
        for j in range(len(timestamps)):
            out_f.write(','.join([str(timestamps[j] * 1000)] +
                                 ['{:.2f}'.format(sum(i.volume[j].values()) / ETH) for i in valuable_infos] +
                                 ['{:.2f}'.format(sum(sum(i.volume[j].values()) for i in other_infos) / ETH)]
                                 ) + '\n')


def save_raw_data(infos: List[ExchangeInfo]):
    with open(INFOS_DUMP, 'wb') as out_f:
        pickle.dump(infos, out_f)


def load_raw_data() -> List[ExchangeInfo]:
    with open(INFOS_DUMP, 'rb') as in_f:
        return pickle.load(in_f)


def save_last_block(block_number: int):
    with open(LAST_BLOCK_DUMP, 'wb') as out_f:
        pickle.dump(block_number, out_f)


def load_last_block() -> int:
    with open(LAST_BLOCK_DUMP, 'rb') as in_f:
        return pickle.load(in_f)


def update_is_required(last_processed_block: int) -> bool:
    return (CURRENT_BLOCK - HISTORY_BEGIN_BLOCK) // HISTORY_CHUNK_SIZE * HISTORY_CHUNK_SIZE + HISTORY_BEGIN_BLOCK > \
           last_processed_block


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if os.path.exists(LAST_BLOCK_DUMP):
        saved_block = load_last_block()
        infos = load_raw_data()
        if update_is_required(saved_block):
            logging.info('Last seen block: {}, current block: {}, loading data for {} blocks...'.format(
                saved_block, CURRENT_BLOCK, CURRENT_BLOCK - saved_block))
            infos = sorted(load_exchange_infos(infos), key=lambda x: x.eth_balance, reverse=True)
            load_logs(saved_block + 1, infos)
            populate_providers(infos)
            populate_roi(infos)
            populate_volume(infos)
            save_last_block(CURRENT_BLOCK)
            save_raw_data(infos)
        else:
            logging.info('Loaded data is up to date')
    else:
        logging.info('Starting from scratch...')
        infos = sorted(load_exchange_infos([]), key=lambda x: x.eth_balance, reverse=True)
        load_logs(HISTORY_BEGIN_BLOCK, infos)
        populate_providers(infos)
        populate_roi(infos)
        populate_volume(infos)
        save_last_block(CURRENT_BLOCK)
        save_raw_data(infos)

    not_empty_infos = [info for info in infos if not is_empty(info)]
    timestamps = load_timestamps()

    save_liquidity_data(infos, timestamps)
    save_total_volume_data(infos, timestamps)

    save_tokens(not_empty_infos, TOKENS_DATA)
    save_providers_data(not_empty_infos)
    save_roi_data(not_empty_infos, timestamps)
    save_volume_data(not_empty_infos, timestamps)


if __name__ == '__main__':
    main()
