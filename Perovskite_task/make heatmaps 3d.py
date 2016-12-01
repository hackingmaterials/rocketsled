import numpy as np
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure(figsize=(12,10))

ax = fig.add_subplot(111,projection='3d')
n = 1000

Alist=[]
Blist=[]
Xlist=[]
scorelist = []


with open('materials_data.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        A = int(row['A'])
        B = int(row['B'])
        X = int(row['anion'])
        score = float(row['complex score'])
        Alist.append(A)
        Blist.append(B)
        Xlist.append(X)
        scorelist.append(score)

colors = cm.afmhot(np.asarray(scorelist)/max(scorelist))

colmap = cm.ScalarMappable(cmap=cm.afmhot)
colmap.set_array(scorelist)

yg = ax.scatter(Alist, Blist, Xlist, alpha=0.15, c=colors, marker='o')
cb = fig.colorbar(colmap)

ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('X')

plt.savefig('3dmap.png')
plt.close()