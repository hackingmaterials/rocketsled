from skopt.dummy_opt import dummy_minimize
from skopt.gp_opt import gp_minimize
import numpy as np

def ABCfun (x):
	A = x[0]
	B = x[1]
	C = x[2]
	D = A*B/C
	return -D


print("done importing")
dimensions = [(1.0,100.0), (1.0, 100.0), (1.0, 100.0)]

dummy_model = dummy_minimize(ABCfun, dimensions, maxiter=1000000)
print("done with dummy")
# gp_model = gp_minimize(ABCfun, dimensions, maxiter=100)

print("dummy:", -dummy_model.fun, "with values", dummy_model.x)
print("dummy avg func vals", np.mean(-dummy_model.func_vals))
# print("GP", -gp_model.fun, "with values", gp_model.x)


