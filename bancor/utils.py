import logging
import time
from typing import Iterable

from config import w3


def get_logs(address: str, topics: Iterable[Iterable[str]], from_block: int = 0, to_block: int = 'latest') -> list:
    return w3.eth.getLogs({
        'fromBlock': from_block,
        'toBlock': to_block,
        'address': address,
        'topics': topics
    })


def timeit(foo):
    def wrapper_foo(*args, **kwargs):
        t = time.time()
        ret = foo(*args, **kwargs)
        t = time.time() - t
        logging.info('{} finished in {:.2f}s'.format(foo.__name__, t))
        return ret

    return wrapper_foo
