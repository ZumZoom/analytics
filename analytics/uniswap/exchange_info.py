class ExchangeInfo:
    def __init__(self,
                 token_address,
                 token_name,
                 token_symbol,
                 token_decimals,
                 exchange_address,
                 eth_balance,
                 token_balance):
        self.token_address = token_address
        self.token_name = token_name
        self.token_symbol = token_symbol
        self.token_decimals = token_decimals
        self.exchange_address = exchange_address
        self.eth_balance = eth_balance
        self.token_balance = token_balance
        self.providers = dict()
        self.history = list()
        self.logs = list()
        self.roi = list()
        self.volume = list()
        self.valuable_traders = list()
