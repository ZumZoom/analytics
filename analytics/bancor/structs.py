from typing import Dict, List


class History:
    def __init__(
        self,
        block_number: int,
        dm_change: float,
        bnt_balance: int,
        token_balance: int,
        trade_volume: int
    ):
        self.block_number: int = block_number
        self.dm_change: float = dm_change
        self.bnt_balance: int = bnt_balance
        self.token_balance: int = token_balance
        self.trade_volume: int = trade_volume


class RelayInfo:
    def __init__(
        self,
        token_address: str,
        token_symbol: str,
        converter_address: str
    ):
        self.token_address: str = token_address
        self.token_symbol: str = token_symbol
        self.underlying_token_symbol: str = None
        self.converter_address: str = converter_address
        self.bnt_balance: int = 0
        self.token_decimals: int = None
        self.token_balance: int = 0
        self.providers: Dict[str, int] = dict()
        self.history: List[History] = list()
        self.converter_logs: List[dict] = list()
        self.relay_logs: List[dict] = list()
