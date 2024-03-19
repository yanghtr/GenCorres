#!/usr/bin/env python
# coding=utf-8
import time
from contextlib import contextmanager

@contextmanager
def timer(task_name):
    t = time.perf_counter()
    try:
        yield
    finally:
        print(task_name, ": ", time.perf_counter() - t, " s.")

def timing(f):
    def wrap(*args, **kwargs):
        t1 = time.perf_counter()
        ret = f(*args, **kwargs)
        t2 = time.perf_counter()
        print('{:s} function took {:.6f} s'.format(f.__name__, (t2 - t1)))
        return ret
    return wrap


if __name__ == '__main__':
    with timer('test'):
        time.sleep(1)
