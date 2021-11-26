from functools import wraps
from time import time

import logging

logger = logging.getLogger()


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        logger.info("%s took %d time to finish" % (f.__name__, elapsed))
        return result

    return wrapper
