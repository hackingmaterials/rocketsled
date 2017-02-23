import numpy as np
from skopt import gbrt_minimize

def f(x):
    print "func was called with arg", x
    return 0

x0 = [[1.5,1.7],[1.2,-1.0]]
y0 = [f(x0[0]), f(x0[1])]
dimensions = [(-2.0, 2.0),(-2.0, 5.0)]

print "beginning optimization"
res = gbrt_minimize(f, dimensions, x0=x0, y0=y0, n_calls=1, n_random_starts=0)
print "next guess to try:", res.x_iters[-1]



#alternatively, using Optimizer
