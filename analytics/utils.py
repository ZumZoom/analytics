import logging
import time


def bytes_to_str(x):
    return x.decode().strip('\x00')


def timeit(foo):
    def wrapper_foo(*args, **kwargs):
        t = time.time()
        ret = foo(*args, **kwargs)
        t = time.time() - t
        logging.info('{} finished in {:.2f}s'.format(foo.__name__, t))
        return ret

    return wrapper_foo
