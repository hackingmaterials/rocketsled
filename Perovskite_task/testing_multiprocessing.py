from multiprocessing import Process, Manager
import os
import multiprocessing
import timeit
import time
import warnings

def worker(procnum, return_list, value_list):
    '''worker function'''
    print "worker_test {}".format(procnum)
    start_time = timeit.default_timer()
    time.sleep(1)
    elapsed = timeit.default_timer() - start_time

    warnings.warn("test warn")
    time.sleep(1)

    return_list.append(procnum)
    value_list.append(elapsed)


if __name__ == '__main__':
    # manager = Manager()
    return_list = Manager().list()
    value_list = Manager().list()
    jobs = []
    for i in range(30):
        warnings.simplefilter("ignore")
        p = multiprocessing.Process(target=worker, args=(i,return_list, value_list))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    time.sleep(1)
    print return_list
    print value_list


