import multiprocessing
import time
import pickle
from pprint import pprint
import os

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


def load_exclusions(filename):
    if os.path.exists(filename):
        with open(filename) as f:
            return pickle.load(f)


exclusions = load_exclusions("excluded_compounds.p")
goldschmidt_rank = load_exclusions("goldschmidt_rank.p")

# # print type(exclusions)
# pprint(exclusions)
# print len(exclusions)
# time.sleep(5)
# pprint(goldschmidt_rank)
# print len(goldschmidt_rank)

for i in xrange(10):
  if i == 5:
    continue
  print i