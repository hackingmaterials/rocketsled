from matplotlib import pyplot as plot
import numpy

# No exclusions
noex = [[0, 230, 464],[0], [0, 171, 447, 780],[0, 60], [0, 46, 268],
[0, 246, 512],[0, 484], [0, 105, 277, 955], [0, 19, 38, 47, 55, 290, 412],
 [0, 279, 971, 1002], [0, 152], [0, 237, 584, 596, 765], [0, 158, 549],
[0, 217, 593, 699],[0, 228, 1081, 1095], [0, 31, 304, 1017],[0, 68, 855, 871, 896, 1024, 1038],[0, 448],
[0, 353, 798],[0, 744] ]

# With exclusions, scoring = 30, exclusions = failures
wex = [[0, 40, 44, 64],[0, 112, 150, 200], [0, 276], [0, 49, 52, 56, 57, 144],
 [0, 453], [0, 52, 131, 281, 286, 316, 333, 387, 402, 418, 447], [0, 297, 336],
  [0, 47, 272],[0], [0, 75, 144, 250], [0, 184], [0, 386], [0, 153],
[0, 37, 95, 175], [0, 413, 439], [0, 22], [0, 99], [0, 201, 234, 243, 344],
 [0, 47, 71, 101, 376] ]

# no exclusions, scoring has 30=failure
volnoex = [[0, 38], [0, 474, 963], [0, 117, 155, 415], [0, 138], [0], [0, 446, 648], [0, 892], [0, 555, 757],
[0, 427, 623, 1091], [0, 139, 162, 376, 649, 771]]

# with exclusions, scoring has 30 = failure, exclusions = failures
volwex  = [[0, 411], [0, 7, 86, 273, 408], [0, 219, 296], [0, 283], [0], [0], [0,66,67], [0], [0, 79], [0,452]]

# no exclusions, scoring has 30 = failure, also reports N = 1 distance candidates as failures
N1noex = [[0, 165, 483], [0, 880], [0, 427, 448], [0, 61, 534, 743], [0, 209, 747], [0, 1056],
[0, 51, 275], [0, 41, 111, 118, 164, 175], [0, 164], [0, 19, 201]]

# no exclusions, scoring has 30 = failure, also reports N = 2 distance candidates as failures
N2noex = [[0, 106], [0, 94, 710], [0, 77, 85, 412], [0], [0, 376, 988], [0, 186] , [0, 107, 365, 408], [0, 641],
[0, 217], [0, 60, 964]]

# chemical rules
chex = [0, 129, 157, 256, 271, 431, 512, 586, 603, 972, 981, 1100, 1165, 1216, 1239, 1258, 1466, 1715, 2258, 3141, 4009]

# combo no exclusions
coex = [[0, 139, 464, 465], [0, 265, 351], [0, 345], [0, 4, 184, 279], [445],
        [0], [0, 318], [0, 63, 73, 119, 343, 377, 467, 471], [0, 154, 177, 411, 473], [0, 110, 215],
        [0, 157, 463], [0, 58, 329, 475, 478], [0, 381], [0, 135, 142, 143, 155, 290, 291, 452], [0, 68, 405],
        [0, 228, 332, 335, 423, 424], [0, 270, 385], [0, 290, 354], [0, 285], [0, 94, 142, 146], [0, 146, 147, 382],
        [0, 183, 196, 303], [0, 109, 113, 275, 279, 399]]

# skopt no exclusions
skex = [[0], [0, 39], [0, 47, 70, 335], [0, 40, 317], [0, 361, 478], [0],
         [0, 97, 184, 486], [0, 319, 467], [0, 244, 263, 329], [0, 192, 201],
         [0, 38], [0], [0, 426, 432], [0, 438], [0, 429], [0, 244, 277], [0, 71], [0, 66], [0, 20, 227, 309],
         [0, 150],[0, 52], [0, 18], [0, 489]]

#round robin
rrnoex = [[0, 29, 125, 687, 703], [0, 470, 502, 598], [0, 202, 808]]

#4000 iteration straight
s4knoex = [[0, 188, 1517, 3128], [0, 81, 397, 1219, 2428], [0, 105, 382, 2538, 2984],
    [0, 179, 722, 1770, 2544, 2750, 2889], [0, 29, 31, 637, 1351, 3042, 3378]]

# responsive sigopt with multiplier of 1.5 and radius 1 to 2690 iterations
rso15_1 = [[0, 121, 2507, 2529, 2543, 2552],[0, 47, 52, 57, 60, 862], [0, 607, 1444, 1448, 1452, 1454, 1601, 2069, 2077],
           [0, 85, 90, 109, 121, 590, 1061, 1191, 1201, 2203, 2206, 2209, 2259],
           [0, 11, 16, 71, 1099, 1102, 1135, 1144, 2471, 2477]]

# incomplete sigopt run m = 1.5 r = 2 to 1600 (test data, not good)
test = [[0, 92, 128, 616, 998], [0, 551, 1026, 1109, 1114, 1147], [0, 65, 78, 111, 193, 415, 435, 1455],
        [0, 87, 941, 982, 1258], [0, 636, 673, 724, 759, 1097, 1106, 1187, 1253, 1301, 1321, 1591]]
testi = [100*x for x in range(17)]

# resposnive sigopt with multiplier of 1.5 and radius 2 to 4000 iterations
rso15_2 = [[0, 543, 2949], [0, 167, 180, 214, 298, 556, 576, 854, 2682, 2716, 2759, 2792],
           [0, 3404, 3509, 3517, 3577, 3581], [0, 76, 238, 320, 324, 355],
           [0, 246, 373, 398, 409, 422, 456, 540, 965, 1338, 1442, 1474, 1525, 1551]]

wexi = [100*x for x in range(6)]
noexi = [100*x for x in range(12)]
rrexi = [100*x for x in range(11)]
s4kexi = [100*x for x in range(41)]
rso15_1i = [100*x for x in range(28)]
rso15_2i = [100*x for x in range(41)]


def stats(biglist, iterations):
    iterlist = []
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


avgwex, stdwex = stats(wex, wexi)
avgnoex, stdnoex = stats(noex, noexi)
avgcoex, stdcoex = stats(coex, wexi)
avgskex, stdskex = stats(skex, wexi)
avgvolnoex, stdvolnoex = stats(volnoex, noexi)
avgvolwex, stdvolwex = stats(volwex, wexi)
avgN1noex, stdN1noex = stats(N1noex, noexi)
avgN2noex, stdN2noex = stats(N2noex, noexi)
avgrrnoex, stdrrnoex = stats(rrnoex, rrexi)
avgs4knoex, stds4knoex = stats(s4knoex, s4kexi)
avgrso15_1, stdrso15_1 = stats(rso15_1, rso15_1i)
avgrso15_2, stdrso15_2 = stats(rso15_2, rso15_2i)

avgtest, stdtest = stats(test, testi)

chexc = list(range(len(chex)))

plot.plot(chex, chexc, 'ro', linestyle='solid')
plot.plot([947*x for x in range(6)], list(range(6)), color='black', linewidth=2)
plot.errorbar(rso15_1i, avgrso15_1, yerr=stdrso15_1, marker='o', color='green')
plot.errorbar(rso15_2i, avgrso15_2, yerr=stdrso15_2, marker='o', color='blue')
# plot.errorbar(testi, avgtest, yerr=stdtest, marker='o', color='blue')
# plot.errorbar(rrexi, avgrrnoex, yerr=stdrrnoex, marker='o', color='blue')
# plot.errorbar(s4kexi, avgs4knoex, yerr=stds4knoex, marker='o', color='green')
# plot.errorbar(noexi, avgnoex, yerr=stdnoex, marker = 'o', color='magenta')
# plot.errorbar(wexi, avgwex, yerr=stdwex, marker='o', color='teal')
# plot.errorbar(wexi, avgcoex, yerr = stdcoex, marker='o', color='orange')
# plot.errorbar(wexi, avgskex, yerr = stdskex, marker='o', color='blue')
# plot.errorbar(noexi, avgN1noex, yerr = stdN1noex, marker='o', color = 'blue')
# plot.errorbar(noexi, avgN2noex, yerr = stdN2noex, marker='o', color='green')
# plot.errorbar(noexri, avgvolnoex, yerr = stdvolnoex, marker='o', color='orange')
plot.xlabel('Iterations')
plot.ylabel('Good Light Splitting Candidates Found')
plot.show()
