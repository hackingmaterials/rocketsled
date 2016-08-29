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

def worker(procnum, return_list, value_list):
    '''worker function'''
    print "worker_test {}".format(procnum)
    output = int(procnum)*2.7
    time.sleep(1)
    print "output: {}".format(output)
    return_list.append(procnum)
    value_list.append(output)


if __name__ == '__main__':
    # manager = Manager()
    return_list = Manager().list()
    value_list = Manager().list()
    jobs = []
    for i in range(30):
        p = multiprocessing.Process(target=worker, args=(i,return_list, value_list))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    time.sleep(1)
    print return_list
    print value_list
    print type(return_list)
    return_list = list(return_list)
    print "new list:", return_list, type(return_list)

