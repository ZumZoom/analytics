import json
from typing import List

from web3._utils.events import get_event_data

from config import w3


class Contract:
    def __init__(self, abi_path, address):
        with open(abi_path) as fh:
            self.contract = w3.eth.contract(abi=json.load(fh), address=address)

    def parse_event(self, event_type: str, event: dict) -> dict:
        return get_event_data(w3.codec, self.contract.events[event_type]._get_event_abi(), event)


class SmartToken(Contract):
    def __init__(self, address):
        super().__init__('abi/SmartToken.abi', address)


class BancorConverterRegistry(Contract):
    def __init__(self, address):
        super().__init__('abi/BancorConverterRegistry.abi', address)

    def get_liquidity_pools(self) -> List[str]:
        return self.contract.functions.getLiquidityPools().call()


class BancorConverter(Contract):
    def __init__(self, address):
        super().__init__('abi/BancorConverter.abi', address)

    def connector_tokens(self, index: int) -> str:
        return self.contract.functions.connectorTokens(index).call()

    def connector_token_count(self) -> int:
        return self.contract.functions.connectorTokenCount().call()

    def reserve_tokens(self, index: int) -> str:
        return self.contract.functions.reserveTokens(index).call()


class ERC20(Contract):
    HARDCODED_INFO = {
        '0xE0B7927c4aF23765Cb51314A0E0521A9645F0E2A': ('DGD', 9),
        '0x8eFFd494eB698cc399AF6231fCcd39E08fd20B15': ('PIX', 0),
        '0x89d24A6b4CcB1B6fAA2625fE562bDD9a23260359': ('SAI', 18),
    }

    def __init__(self, address):
        super().__init__('abi/ERC20.abi', address)

    def decimals(self) -> int:
        if self.contract.address in self.HARDCODED_INFO:
            return self.HARDCODED_INFO[self.contract.address][1]
        else:
            try:
                return self.contract.functions.decimals().call()
            except Exception:
                with open('abi/ERC20_CAPS.abi') as fh:
                    token = w3.eth.contract(abi=json.load(fh), address=self.contract.address)
                return token.functions.DECIMALS().call()

    def symbol(self) -> str:
        if self.contract.address in self.HARDCODED_INFO:
            return self.HARDCODED_INFO[self.contract.address][0]
        else:
            try:
                return self.contract.functions.symbol().call()
            except Exception:
                try:
                    with open('abi/ERC20_CAPS.abi') as fh:
                        token = w3.eth.contract(abi=json.load(fh), address=self.contract.address)
                    return token.functions.SYMBOL().call()
                except Exception:
                    try:
                        with open('abi/ERC20_bytes.abi') as fh:
                            token = w3.eth.contract(abi=json.load(fh), address=self.contract.address)
                        return token.functions.symbol().call().decode().strip('\x00')
                    except Exception:
                        return None
