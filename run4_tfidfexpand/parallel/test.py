import multiprocessing as mp
from multiprocessing import Pool


def foo(x):
    print(mp.current_process().pid)
    return x



pool = Pool(2)
pool.map(foo, range(2))
pool.close()
pool.join()
