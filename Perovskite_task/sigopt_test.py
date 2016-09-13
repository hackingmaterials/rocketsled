from sigopt import Connection
from pprint import pprint
import time
import numpy as np
import matplotlib.pyplot as plt

num = 15

conn = Connection(client_token="DVTOCGKXJPFHSMSGMFWNAIZOXRGBVIZBDVHOTMRRIKKXHAKK")

def runfun(x,y,z):
    # bound are (1-10) for x, y and z
    return x*y-(5*z)+5

experiment = conn.experiments().create(
    name='Alexs New Test Optimization',
    parameters=[
        dict(name='x', type='int', bounds=dict(min=1, max=2)),
        dict(name='y', type='int', bounds=dict(min=1, max=2)),
        dict(name='z', type='int', bounds=dict(min=1, max=2))],
)
print("Created experiment: https://sigopt.com/experiment/" + experiment.id)

def evaluate_model(assignments):
    x = assignments['x']
    y = assignments['y']
    z = assignments['z']
    return runfun(x,y,z)

# Run the Optimization Loop between 10x - 20x the number of parameters
times=[]
values=[]
for i in range(num):
    start_time =time.time()
    suggestion = conn.experiments(experiment.id).suggestions().create()
    value = evaluate_model(suggestion.assignments)
    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        value=value,
    )
    print suggestion.assignments
    elapsed = time.time() - start_time
    times.append(elapsed)
    values.append(value)
    print "CALCULATION", i+1, "WITH SCORE", value


timeplot = plt.figure(1)
timeline = plt.plot(list(range(num)), times)
plt.xlabel("Iteration")
plt.ylabel("Computational time and Internet Latency (s)")


scoreplot = plt.figure(2)
scoreline = plt.plot(list(range(num)), values)
plt.xlabel("Iterations")
plt.ylabel("Best Score (out of 100)")
plt.show()
# suggestion = conn.experiments(experiment.id).suggestions().create()
# pprint(suggestion.assignments)
# print type(suggestion.assignments)
# print suggestion.assignments['x']

































