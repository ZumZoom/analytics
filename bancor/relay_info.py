from typing import List

from history import History


class RelayInfo:
    def __init__(self,
                 token_address,
                 token_symbol,
                 token_decimals,
                 converter_address):
        self.token_address = token_address
        self.token_symbol = token_symbol
        self.token_decimals = token_decimals
        self.converter_address = converter_address
        self.bnt_balance = 0
        self.token_balance = 0
        self.providers = dict()
        self.history: List[History] = list()
        self.converter_logs: List[dict] = list()
        self.relay_logs: List[dict] = list()
        self.valuable_traders: List[str] = list()
