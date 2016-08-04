# from turboworks.dummy_opt import dummy_minimize
#
# name_index = ['Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
#               'Zn', 'Ga', 'Ge',
#               'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba',
#               'La', 'Hf',
#               'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
#
# anion_names = ['O3','O2N','ON2','N3','O2F','OFN','O2S']
#
# initial_guess = dummy_minimize([name_index, name_index, anion_names])
#
# print initial_guess

import numpy as np
import datetime


def get_cand_stats(cands, iters):
    max_cand = 0
    for cand in cands:
        if cand[-1] > max_cand:
            max_cand = cand[-1]

    avg_iterations_at_candidate = []
    std_iterations_at_candidate = []
    for i in range(max_cand):
        temp = []
        for iter_set in iters:
            try:
                temp.append(iter_set[i])
            except:
                pass
        avg_iterations_at_candidate.append(np.asarray(temp).mean())
        std_iterations_at_candidate.append(np.asarray(temp).std())

    zero = [0]
    avg_iterations_at_candidate = zero + avg_iterations_at_candidate
    std_iterations_at_candidate = zero + std_iterations_at_candidate

    all_cands = list(range(max_cand + 1))
    return avg_iterations_at_candidate, std_iterations_at_candidate, all_cands


skopt_cands = [[1,2,3],[1,2,3,4,5,6],[1,2,3]]
skopt_iters = [[23, 333, 500], [12, 355, 485, 605,700,912], [99, 294, 540]]
combo_cands = [[1,2,3,4],[1,2,3,4,5,6,7],[1,2,3,4,5,6,7]]
combo_iters = [[66, 126, 355], [98, 123, 345, 567, 677, 723, 923], [140, 195, 299, 399, 455, 788, 977]]

skopt_iters, skopt_iters_std, skopt_cands = get_cand_stats(skopt_cands, skopt_iters)
combo_iters, combo_iters_std, combo_cands = get_cand_stats(combo_cands, combo_iters)

print "skopt iterations:", skopt_iters
print "skopt iteration std.:", skopt_iters_std
print "skopt candidates:", skopt_cands


'''Save Results'''
text_file = open('results.txt', 'w')
text_file.write("\n TIME OF RUN: {} \n".format(datetime.datetime.now().time().isoformat()))
text_file.write("skopt iterations: {} \n".format(skopt_iters))
text_file.write("skopt std dev iterations: {} \n".format(skopt_iters_std))
text_file.write("skopt candidate list: {} \n".format(skopt_cands))
text_file.write("combo iterations: {} \n".format(combo_iters))
text_file.write("combo std dev iterations: {} \n".format(combo_iters_std))
text_file.write("combo candidate list: {} \n".format(combo_cands))

import matplotlib.pyplot as plt

candplot = plt.figure(1)
skopterr = plt.errorbar(skopt_iters, skopt_cands, xerr=skopt_iters_std, fmt='og', ecolor='black',
                        capthick=2, capsize=3, elinewidth=2)
skoptline = plt.plot(skopt_iters, skopt_cands, 'g')

comboerr = plt.errorbar(combo_iters, combo_cands, xerr=combo_iters_std, fmt='ob', ecolor='black',
                        capthick=2, capsize=3, elinewidth=2)

comboline = plt.plot(combo_iters, combo_cands, 'b')

rand_iters=[]
if combo_iters[-1] > skopt_iters[-1]:
    rand_iters = combo_iters
else:
    rand_iters = skopt_iters
randline = plt.plot(rand_iters, [i/946.4 for i in rand_iters])
plt.setp(skoptline, linewidth=3, color='g')
plt.setp(comboline, linewidth=3, color='b')
plt.setp(randline, linewidth=3, color='black')
plt.xlabel("Iterations")
plt.ylabel("Candidates Found")
plt.title("Candidates vs Iterations")

plt.show()