class History:
    def __init__(self, block_number, dm_change, bnt_balance, token_balance, trade_volume, volume_by_trader):
        self.block_number = block_number
        self.dm_change = dm_change
        self.bnt_balance = bnt_balance
        self.token_balance = token_balance
        self.trade_volume = trade_volume
        self.volume_by_trader = volume_by_trader
