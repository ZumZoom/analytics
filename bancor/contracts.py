import json
from typing import List, Dict

from web3._utils.events import get_event_data

from config import w3
from utils import timeit


class Contract:
    def __init__(self, abi_path, address):
        with open(abi_path) as fh:
            self.contract = w3.eth.contract(abi=json.load(fh), address=address)

    def parse_event(self, event_type: str, event: dict) -> dict:
        return get_event_data(self.contract.events[event_type]._get_event_abi(), event)


class SmartToken(Contract):
    pass


class BancorConverterRegistry(Contract):
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

    @timeit
    def all_converter_addresses(self, tokens: List[str] = None) -> Dict[str, List[str]]:
        return {
            token: [self.converter_address(token, i) for i in range(self.converter_count(token))]
            for token in (tokens if tokens else self.all_tokens())
        }

    @timeit
    def latest_converter_addresses(self, tokens: List[str] = None) -> Dict[str, str]:
        return {
            token: self.converter_address(token, self.converter_count(token) - 1)
            for token in (tokens if tokens else self.all_tokens())
        }


class BancorConverter(Contract):
    pass
