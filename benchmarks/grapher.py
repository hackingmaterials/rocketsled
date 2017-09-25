import plotly.plotly as py
import plotly.graph_objs as go
import pickle

# Create random data with numpy
import numpy as np


def depickle(file):
    return pickle.load(open(file, 'rb'))


py.sign_in('ardunn', '7CM5eLh5l1vMqIwqvlKE') # Replace the username, and API key with your credentials.


BBFUN = "rosen"
y_gp = depickle('gp_{}.pickle'.format(BBFUN))
y_gbt = depickle('gbt_{}.pickle'.format(BBFUN))
y_rf = depickle('rf_{}.pickle'.format(BBFUN))
y_ran = depickle('ran_{}.pickle'.format(BBFUN))
x = list(range(1000))

width = 2

# Create a trace
gp = go.Scatter(x = x, y = y_gp, line={'color': 'green', 'width': width})

gbt = go.Scatter(x = x, y = y_gbt, line={'color': 'blue', 'width': width})

rf = go.Scatter(x = x, y = y_rf, line={'color': 'red', 'width': width})

ran = go.Scatter(x = x, y = y_ran, line={'color': 'black', 'width': width})

data = [gp, gbt, rf, ran]

py.plot(data, filename='{}'.format(BBFUN))