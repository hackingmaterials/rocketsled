import multiprocessing
import time

jobs = []

def goodfun():
    print "This is goodfun"
    time.sleep(1)

def mysteryfun():
    print "This is mysteryfun"
    time.sleep(1)

def fun1(text):
    print text + " running"
    goodfun()
    print text + " ran"


def fun2(text):
    print text + " running"
    mysteryfun()
    print text + " ran"


p1 = multiprocessing.Process(target=fun1, args=("goodfun",))
p2 = multiprocessing.Process(target=fun2, args=("mysteryfun",))
jobs.append(p1)
jobs.append(p2)

for proc in jobs:
    proc.start()

for proc in jobs:
    proc.join()

print "done"