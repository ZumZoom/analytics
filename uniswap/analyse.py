import json
import logging
import os
import pickle
from collections import defaultdict
from math import sqrt

from retrying import retry
from typing import List, Iterable

from web3.utils.events import get_event_data

from config import uniswap_factory, pool, web3, UNISWAP_EXCHANGE_ABI, STR_ERC_20_ABI, HARDCODED_INFO, \
    STR_CAPS_ERC_20_ABI, ERC_20_ABI, HISTORY_BEGIN_BLOCK, CURRENT_BLOCK, HISTORY_CHUNK_SIZE, ETH, UNISWAP_BEGIN_BLOCK, \
    LOGS_BLOCKS_CHUNK, LIQUIDITY_DATA, PROVIDERS_DATA, TOKENS_DATA, INFOS_DUMP, LAST_BLOCK_DUMP, \
    ALL_EVENTS, EVENT_TRANSFER, EVENT_ADD_LIQUIDITY, EVENT_REMOVE_LIQUIDITY, EVENT_ETH_PURCHASE, ROI_DATA, \
    EVENT_TOKEN_PURCHASE, VOLUME_DATA
from exchange_info import ExchangeInfo
from roi_info import RoiInfo
from utils import timeit, bytes_to_str


@timeit
def load_token_count() -> int:
    return uniswap_factory.functions.tokenCount().call()


@timeit
def load_tokens(token_count: int) -> List[str]:
    if not token_count:
        token_count = load_token_count()
    tokens = pool.map(lambda i: uniswap_factory.functions.getTokenWithId(i).call(), range(1, token_count + 1))
    logging.info('Found {} tokens'.format(len(tokens)))
    return tokens


@timeit
def load_exchanges(tokens: List[str]) -> List[str]:
    if not tokens:
        tokens = load_tokens()
    exchanges = pool.map(lambda t: uniswap_factory.functions.getExchange(t).call(), tokens)
    logging.info('Found {} exchanges'.format(len(exchanges)))
    return exchanges


def load_exchange_data_impl(token_address, exchange_address):
    exchange = web3.eth.contract(abi=UNISWAP_EXCHANGE_ABI, address=exchange_address)
    token = web3.eth.contract(abi=STR_ERC_20_ABI, address=token_address)
    if token_address in HARDCODED_INFO:
        token_name, token_symbol, token_decimals = HARDCODED_INFO[token_address]
    else:
        try:
            token_name = token.functions.name().call()
            token_symbol = token.functions.symbol().call()
            token_decimals = token.functions.decimals().call()
        except:
            try:
                token = web3.eth.contract(abi=STR_CAPS_ERC_20_ABI, address=token_address)
                token_name = token.functions.NAME().call()
                token_symbol = token.functions.SYMBOL().call()
                token_decimals = token.functions.DECIMALS().call()
            except:
                try:
                    token = web3.eth.contract(abi=ERC_20_ABI, address=token_address)
                    token_name = bytes_to_str(token.functions.name().call())
                    token_symbol = bytes_to_str(token.functions.symbol().call())
                    token_decimals = token.functions.decimals().call()
                except:
                    logging.warning('FUCKED UP {}'.format(token_address))
                    return None

    token_balance = token.functions.balanceOf(exchange.address).call(block_identifier=CURRENT_BLOCK)
    eth_balance = web3.eth.getBalance(exchange.address, block_identifier=CURRENT_BLOCK)
    return ExchangeInfo(token.address,
                        token_name,
                        token_symbol,
                        token_decimals,
                        exchange.address,
                        eth_balance,
                        token_balance)


@timeit
def load_exchange_infos(infos: List[ExchangeInfo]) -> List[ExchangeInfo]:
    token_count = load_token_count()
    tokens = load_tokens(token_count)
    exchanges = load_exchanges(tokens)

    new_infos = [info for info in pool.starmap(load_exchange_data_impl, zip(tokens, exchanges)) if info]
    if infos:
        known_tokens = dict((info.token_address, info) for info in infos)
        for new_info in new_infos:
            info = known_tokens.get(new_info.token_address)
            if info:
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
    return pool.map(lambda n: web3.eth.getBlock(n)['timestamp'], get_chart_range())


def get_logs(address: str, topics: List, start_block: int) -> List:
    @retry(stop_max_attempt_number=3, wait_fixed=1)
    def get_chunk(start):
        return web3.eth.getLogs({
            'fromBlock': start,
            'toBlock': min(start + LOGS_BLOCKS_CHUNK, CURRENT_BLOCK),
            'address': address,
            'topics': topics})

    log_chunks = pool.map(get_chunk, range(start_block, CURRENT_BLOCK, LOGS_BLOCKS_CHUNK))
    return [log for chunk in log_chunks for log in chunk]


@timeit
def load_logs(infos: List[ExchangeInfo]) -> List[ExchangeInfo]:
    for info in infos:
        if info.logs:
            block_number_to_check = info.logs[-1]['blockNumber'] + 1
        else:
            block_number_to_check = UNISWAP_BEGIN_BLOCK
        exchange = web3.eth.contract(abi=UNISWAP_EXCHANGE_ABI, address=info.exchange_address)
        new_logs = get_logs(exchange.address, [ALL_EVENTS], block_number_to_check)
        info.logs += new_logs

    logging.info('Loaded transfer logs for {} exchanges'.format(len(infos)))
    return infos


@timeit
def populate_providers(infos: List[ExchangeInfo]) -> List[ExchangeInfo]:
    for info in infos:
        exchange = web3.eth.contract(abi=UNISWAP_EXCHANGE_ABI, address=info.exchange_address)
        info.providers = defaultdict(int)
        for log in info.logs:
            if log['topics'][0].hex() != EVENT_TRANSFER:
                continue
            event = get_event_data(exchange.events.Transfer._get_event_abi(), log)
            if event['args']['from'] == '0x0000000000000000000000000000000000000000':
                info.providers[event['args']['to']] += event['args']['value']
            elif event['args']['to'] == '0x0000000000000000000000000000000000000000':
                info.providers[event['args']['from']] -= event['args']['value']
            else:
                info.providers[event['args']['from']] -= event['args']['value']
                info.providers[event['args']['to']] += event['args']['value']
    logging.info('Loaded info about providers of {} exchanges'.format(len(infos)))
    return infos


@timeit
def populate_roi(infos: List[ExchangeInfo]) -> List[ExchangeInfo]:
    for info in infos:
        info.roi = list()
        exchange = web3.eth.contract(abi=UNISWAP_EXCHANGE_ABI, address=info.exchange_address)
        i = 0
        eth_balance, token_balance = 0, 0
        for block_number in get_chart_range():
            dm_numerator, dm_denominator, trade_volume = 1, 1, 0
            while i < len(info.logs) and info.logs[i]['blockNumber'] < block_number:
                log = info.logs[i]
                i += 1
                topic = log['topics'][0].hex()
                if topic == EVENT_TRANSFER:
                    continue
                elif topic == EVENT_ADD_LIQUIDITY:
                    event = get_event_data(exchange.events.AddLiquidity._get_event_abi(), log)
                    eth_balance += event['args']['eth_amount']
                    token_balance += event['args']['token_amount']
                elif topic == EVENT_REMOVE_LIQUIDITY:
                    event = get_event_data(exchange.events.RemoveLiquidity._get_event_abi(), log)
                    eth_balance -= event['args']['eth_amount']
                    token_balance -= event['args']['token_amount']
                elif topic == EVENT_ETH_PURCHASE:
                    event = get_event_data(exchange.events.EthPurchase._get_event_abi(), log)
                    eth_new_balance = eth_balance - event['args']['eth_bought']
                    token_new_balance = token_balance + event['args']['tokens_sold']
                    dm_numerator *= eth_new_balance * token_new_balance
                    dm_denominator *= eth_balance * token_balance
                    trade_volume += event['args']['eth_bought'] / 0.997
                    eth_balance = eth_new_balance
                    token_balance = token_new_balance
                else:
                    event = get_event_data(exchange.events.TokenPurchase._get_event_abi(), log)
                    eth_new_balance = eth_balance + event['args']['eth_sold']
                    token_new_balance = token_balance - event['args']['tokens_bought']
                    dm_numerator *= eth_new_balance * token_new_balance
                    dm_denominator *= eth_balance * token_balance
                    trade_volume += event['args']['eth_sold']
                    eth_balance = eth_new_balance
                    token_balance = token_new_balance

            try:
                info.roi.append(RoiInfo(sqrt(dm_numerator / dm_denominator), eth_balance, token_balance, trade_volume))
            except ValueError:
                print(info.token_symbol, info.exchange_address)

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
                    event = get_event_data(exchange.events.EthPurchase._get_event_abi(), log)
                    trade_volume[event['args']['buyer']] += event['args']['eth_bought'] / 0.997
                    total_trade_volume[event['args']['buyer']] += event['args']['eth_bought'] / 0.997
                elif topic == EVENT_TOKEN_PURCHASE:
                    event = get_event_data(exchange.events.TokenPurchase._get_event_abi(), log)
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
    return info.eth_balance >= 10 * ETH


@timeit
def populate_liquidity_history(infos: List[ExchangeInfo]) -> List[ExchangeInfo]:
    for info in infos:
        exchange = web3.eth.contract(abi=UNISWAP_EXCHANGE_ABI, address=info.exchange_address)
        history_len = len(info.history)
        new_history = pool.map(lambda block_number: web3.eth.getBalance(exchange.address, block_number) / ETH,
                               get_chart_range(HISTORY_BEGIN_BLOCK + history_len * HISTORY_CHUNK_SIZE))
        info.history += new_history

    print('Loaded history of balances of {} exchanges'.format(len(infos)))
    return infos


def save_tokens(infos: List[ExchangeInfo]):
    with open(TOKENS_DATA, 'w') as out_f:
        json.dump({'results': [{'id': info.token_symbol.lower(), 'text': info.token_symbol} for info in infos]},
                  out_f,
                  indent=1)


def save_liquidity_data(infos: List[ExchangeInfo]):
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
        with open(PROVIDERS_DATA.format(info.token_symbol.lower()), 'w') as out_f:
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


def save_roi_data(infos: List[ExchangeInfo]):
    timestamps = load_timestamps()

    for info in infos:
        with open(ROI_DATA.format(info.token_symbol.lower()), 'w') as out_f:
            out_f.write('timestamp,ROI,Token Price,Trade Volume\n')
            for j in range(len(timestamps)):
                if info.roi[j].eth_balance == 0:
                    continue
                out_f.write(','.join([str(timestamps[j] * 1000),
                                      '{}'.format(info.roi[j].dm_change),
                                      '{}'.format(info.roi[j].token_balance / info.roi[j].eth_balance),
                                      '{:.2f}'.format(info.roi[j].trade_volume / ETH)]) + '\n')


def save_volume_data(infos: List[ExchangeInfo]):
    timestamps = load_timestamps()

    for info in infos:
        with open(VOLUME_DATA.format(info.token_symbol.lower()), 'w') as out_f:
            out_f.write(','.join(['timestamp'] + ['\u200b{}'.format(t) for t in info.valuable_traders] + ['Other']) + '\n')
            for j in range(len(timestamps)):
                out_f.write(','.join([str(timestamps[j] * 1000)] +
                                     ['{:.2f}'.format(info.volume[j][t] / ETH) if info.volume[j][t] else ''
                                      for t in info.valuable_traders + ['Other']]) + '\n')


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if os.path.exists(LAST_BLOCK_DUMP):
        saved_block = load_last_block()
        infos = load_raw_data()
        if saved_block + 1000 < CURRENT_BLOCK:
            infos = sorted(load_exchange_infos(infos), key=lambda x: x.eth_balance, reverse=True)
            load_logs(infos)
            populate_liquidity_history(infos)
            populate_providers(infos)
            populate_roi(infos)
            populate_volume(infos)
            save_last_block(CURRENT_BLOCK)
            save_raw_data(infos)
        else:
            logging.info('Loaded data is up to date')
    else:
        infos = sorted(load_exchange_infos([]), key=lambda x: x.eth_balance, reverse=True)
        load_logs(infos)
        populate_liquidity_history(infos)
        populate_providers(infos)
        populate_roi(infos)
        populate_volume(infos)
        save_last_block(CURRENT_BLOCK)
        save_raw_data(infos)

    valuable_infos = [info for info in infos if is_valuable(info)]
    save_tokens(valuable_infos)
    save_liquidity_data(infos)
    save_providers_data(valuable_infos)
    save_roi_data(valuable_infos)
    save_volume_data(valuable_infos)
