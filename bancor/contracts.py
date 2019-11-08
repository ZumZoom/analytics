import json
from typing import List, Dict

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

    def token_count(self) -> int:
        return self.contract.functions.tokenCount().call()

    def converter_count(self, token: str) -> int:
        return self.contract.functions.converterCount(token).call()

    def tokens(self, index: int) -> str:
        return self.contract.functions.tokens(index).call()

    def all_tokens(self, count: int = None) -> List[str]:
        return [self.tokens(i) for i in range(count if count else self.token_count())]

    def converter_address(self, token: str, index: int) -> str:
        return self.contract.functions.converterAddress(token, index).call()

    def all_converter_addresses(self, tokens: List[str] = None) -> Dict[str, List[str]]:
        return {
            token: [self.converter_address(token, i) for i in range(self.converter_count(token))]
            for token in (tokens if tokens else self.all_tokens())
        }

    def latest_converter_addresses(self, tokens: List[str] = None) -> Dict[str, str]:
        ret = dict()
        for token in (tokens if tokens else self.all_tokens()):
            converter_count = self.converter_count(token)
            if converter_count:
                ret[token] = self.converter_address(token, converter_count - 1)
        return ret


class BancorConverter(Contract):
    def __init__(self, address):
        super().__init__('abi/BancorConverter.abi', address)

    def connector_tokens(self, index: int) -> str:
        return self.contract.functions.connectorTokens(index).call()

    def reserve_tokens(self, index: int) -> str:
        return self.contract.functions.reserveTokens(index).call()


class ERC20(Contract):
    def __init__(self, address):
        super().__init__('abi/ERC20.abi', address)

    def decimals(self) -> int:
        return self.contract.functions.decimals().call()

    def symbol(self) -> str:
        return self.contract.functions.symbol().call().strip(b'\x00').decode()
