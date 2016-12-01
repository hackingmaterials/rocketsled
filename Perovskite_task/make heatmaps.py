import matplotlib.pyplot as plt
import numpy as np
import csv
from perovskite_task import mendeleev_rank2anion_name, eneg_rank2anion_name
from mpl_toolkits.axes_grid1 import make_axes_locatable

e=1.5

def gettrackers(Xchoice, filename='mendeleev_tracker.csv'):
    trackers = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        tracker = []
        for row in reader:
            A = int(row['A'])
            B = int(row['B'])
            X = int(row['anion'])
            if X == Xchoice:
                tracker = [A,B,X]
                trackers.append(tracker)
    return trackers

def writemap(Xchoice, filename='materials_data.csv'):
    hm = np.zeros((52, 52))
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            A = int(row['A'])
            B = int(row['B'])
            X = int(row['anion'])
            cs = float(row['complex score'])
            ss = int(row['simple score'])
            cp = float(row['complex product'])
            # selecting a scoring algorithm
            score = cs
            if X == Xchoice:
                if score == 30:
                    score *= e
                hm[A, B] = score
        return hm

if __name__ == "__main__":

    Xchoice = 3
    hm = writemap(Xchoice)
    tr = gettrackers(Xchoice, 'mendeleev_trackerO3.csv')
    for index, data in enumerate(tr):
        if index!=0:
            hm[data[0],data[1]] = 30*e*0.87
        print "saving image", index
        im = plt.imshow(hm, cmap='hot', interpolation='nearest', origin='lower', vmin=0, vmax=e * 30)
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "5%", pad="3%")
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig('hmcomplex3_{}.png'.format(str(index).zfill(5)), bbox_inches='tight')
        plt.close()