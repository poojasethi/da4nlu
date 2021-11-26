from functools import wraps
from time import time

import logging
import sys


logger = logging.getLogger()


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        logger.info("%s took %d seconds to finish" % (f.__name__, elapsed))
        return result

    return wrapper
