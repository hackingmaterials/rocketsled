
from turboworks.reference import ref_dict

from turboworks.gp_opt import gp_minimize


import collections

def update(d, u):
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d



from functools import reduce  # forward compatibility for Python 3
import operator

def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value

input = [[1,2,3],[4,5,6]]
output = [12, 16]
dim = [(1,100), (1,100)]
y = gp_minimize(input, output, dim)
print y