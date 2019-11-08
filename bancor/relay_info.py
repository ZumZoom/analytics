from typing import List

from history import History


class RelayInfo:
    def __init__(self,
                 token_address,
                 token_symbol,
                 converter_address):
        self.token_address = token_address
        self.token_symbol = token_symbol
        self.underlying_token_symbol = None
        self.converter_address = converter_address
        self.bnt_balance = 0
        self.token_decimals = 0
        self.token_balance = 0
        self.providers = dict()
        self.history: List[History] = list()
        self.converter_logs: List[dict] = list()
        self.relay_logs: List[dict] = list()
