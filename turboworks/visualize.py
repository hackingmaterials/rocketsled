from bokeh.plotting import figure, output_file, show
import pylab as plt
import numpy as np
from turboworks.db import DB
from bokeh.plotting import figure, curdoc
from bokeh.models.sources import ColumnDataSource
from bokeh.client import push_session
from bokeh.driving import linear
import subprocess


# class Visualize(object):

def visualize(pause=1.0, threshold=None):
    db = DB()

    X = []
    Y = []
    plt.ion()
    graph = plt.plot(X, Y)[0]



    while True:

        try:
            min = db.min.value
            data = db.min.data
        except (ValueError):
            plt.pause(pause)
            continue


        Y.append(min)
        print "data:", data
        print "ydata:", Y
        graph.set_ydata(Y)
        graph.set_xdata(range(len(Y)))
        plt.draw()
        plt.pause(pause)





# if __name__ == "__main__":
#     viz = Visualize()

