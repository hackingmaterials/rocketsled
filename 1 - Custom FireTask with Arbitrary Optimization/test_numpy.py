import numpy as np

A = [1.1,2.3,3,4,5]
B = [2,3,4,5,1]

C = np.divide(A,B)
dimC = np.shape(C)
print dimC[0]
print np.random.rand(1,dimC[0])
print A