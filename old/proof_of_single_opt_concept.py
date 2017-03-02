import numpy as np
from skopt import gbrt_minimize

x0 = [[1.5,1.7],[1.2,-1.0]]
y0 = [3.4,5.7]
dimensions = [(-2.0, 2.0),(-2.0, 5.0)]

print "beginning optimization"
res = gbrt_minimize(lambda x:0, dimensions, x0=x0, y0=y0, n_calls=1, n_random_starts=0)
print "next guess to try:", res.x_iters[-1]



#alternatively, using Optimizer
