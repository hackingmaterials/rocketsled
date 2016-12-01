import matplotlib.pyplot as plot
import numpy

noex = [[0, 148, 186, 1138], [0, 5, 9, 252, 1150, 1193], [0, 6], [0, 28, 37, 95, 113, 121, 622, 665],
         [0, 5, 8, 19, 26, 37, 966], [0, 533, 569, 642], [0, 334, 356, 411, 780, 814], [0, 9, 895],
         [0, 8, 35, 39, 44, 48, 668], [0, 13, 23, 44, 197], [0, 6, 9, 10, 27, 32, 580], [0, 7, 11, 16, 21],
         [0, 117, 146], [0, 162], [0, 337], [0, 68], [0], [0, 58, 641, 662, 687, 914], [0, 295, 547], [0, 670]]

nx = [3, 5, 1, 7, 6, 3, 5, 2, 6, 4, 6, 4, 2 ,1, 1, 1 ,0 ,5 ,2 ,1]

wex = [[0, 26, 41, 232, 440, 482], [0, 233, 348], [0, 7, 12, 15, 54, 231, 465], [0, 102, 199, 435],
       [0, 4, 153, 180, 273], [0, 3, 173, 175], [0, 11, 12, 30], [0, 4, 5, 15, 204, 483],
       [0, 3, 4, 8, 17, 56, 193, 478], [0, 2, 4, 9, 114, 131], [0, 67, 130], [0, 12, 16, 167], [0, 272], [0],
        [0, 100, 131, 282], [0, 5, 10, 13, 16, 28, 42, 392], [0, 3, 5, 6, 7, 9, 12, 18, 22, 98, 420],
        [0, 3, 4, 6, 7, 9, 10, 18, 247], [0, 189, 415], [0, 6, 10, 22, 45, 114, 121]]

wx = [5, 2, 6, 3, 4, 3, 3, 5, 7, 5, 2 ,3 ,1 ,0, 3, 7, 10, 8, 2, 6]

iterations = [50*x for x in range(25)]
iterlist = []
def stats(biglist):
    for w in biglist:
        iter1 = [-1 for x in range(len(iterations))]
        for i in w:
            for j, val in enumerate(iterations):
                if i<=val:
                    iter1[j] = iter1[j] + 1

        iterlist.append(iter1)

    avg = numpy.mean(iterlist, axis=0)
    std = numpy.std(iterlist, axis=0)
    return avg, std

avgnoex, stdnoex = stats(noex)
avgwex, stdwex = stats(wex)

plot.errorbar(iterations, avgnoex, yerr=stdnoex)
plot.errorbar(iterations, avgwex, yerr=stdwex)
# plot.show()


dim = []
ranges = [(0,12),(12,25),(25,38),(38,51)]
for a in ranges:
    for b in ranges:
        dim.append([a,b,(0,7)])

print dim
print len(dim)






