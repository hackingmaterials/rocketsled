from multiprocessing import Process, Manager
import os
import multiprocessing
import time

# def myfun (num):
#     print "running process #{}".format(num)
#     time.sleep(5)
#     print "finished process #{}".format(num)
#     return 12
#
#
# if __name__ == '__main__':
#     p1 = Process(target=myfun, args=(1,))
#     p2 = Process(target=myfun, args=(2,))
#     p1.start()
#     p2.start()
#     p1.join()
#     print "skrr"
#
#

def worker(procnum, return_dict):
    '''worker function'''
    return_list.append(procnum)


if __name__ == '__main__':
    manager = Manager()
    return_list = manager.list()
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i,return_list))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
    print return_list