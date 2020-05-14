from typing import Dict, List


class RoiInfo:
    def __init__(
        self,
        dm_change: float,
        eth_balance: int,
        token_balance: int,
        trade_volume: int
    ):
        self.dm_change: float = dm_change
        self.eth_balance: int = eth_balance
        self.token_balance: int = token_balance
        self.trade_volume: int = trade_volume


class ExchangeInfo:
    def __init__(
        self,
        token_address: str,
        token_name: str,
        token_symbol: str,
        token_decimals: int,
        exchange_address: str,
        eth_balance: int,
        token_balance: int
    ):
        self.token_address: str = token_address
        self.token_name: str = token_name
        self.token_symbol: str = token_symbol
        self.token_decimals: int = token_decimals
        self.exchange_address: str = exchange_address
        self.eth_balance: int = eth_balance
        self.token_balance: int = token_balance
        self.providers: Dict[str, int] = dict()
        self.history: List[int] = list()
        self.logs: List[dict] = list()
        self.roi: List[RoiInfo] = list()
        self.volume: List[Dict[str, int]] = list()
        self.valuable_traders: List[str] = list()
