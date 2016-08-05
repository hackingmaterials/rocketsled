from pymongo import MongoClient
from pprint import pprint
from turboworks.gp_opt import gp_minimize
import math
import numpy as np
import combo
from turboworks.discrete_spacify import calculate_discrete_space
from turboworks.dummy_opt import dummy_minimize
import matplotlib.pyplot as plt
import datetime

connection = MongoClient()
unc = connection.unc.data_raw

"""
This file is doing perovskite testing without using the TW/FW overhead
"""

# STATIC DATA
GOOD_CANDS_LS = [(3, 23, 0), (11, 51, 0), (12, 73, 1), (20, 32, 0), (20, 50, 0), (20, 73, 1), (38, 32, 0), (38, 50, 0),
                 (38, 73, 1), (39, 73, 2), (47, 41, 0), (50, 22, 0), (55, 41, 0), (56, 31, 4), (56, 49, 4), (56, 50, 0),
                 (56, 73, 1), (57, 22, 1), (57, 73, 2), (82, 31, 4)]  # LIGHT SPLITTERS (20)
GOOD_CANDS_OS = [(20, 50, 0), (37, 22, 4), (37, 41, 0), (38, 22, 0), (38, 31, 4), (38, 50, 0), (55, 73, 0),
                 (56, 49, 4)]  # OXIDE SHIELDS (8)
NUM_CANDS = 18928
"""
* Main index ranks the elements based on their atomic number or mendeleev number(continuous)
* Atomic index is the elements as atomic number
* Name index is the names of the elements
* Mendeleev index is the elements as Mendeleev style number
"""
main_index = list(range(52))
atomic_index = [3, 4, 5, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40,
         41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
name_index = ['Li','Be','B','Na','Mg','Al','Si','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge',
              'As','Rb','Sr','Y','Zr','Nb','Mo','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','Cs','Ba','La','Hf',
              'Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi']
mendeleev_index = [1,67,72,2,68,73,78,3,7,11,43,46,49,52,55,58,61,64,69,74,79,84,4,8,12,44,47,50,56,59,62,65,70,75,80,
                   85,90,5,9,13,45,48,51,54,57,60,63,66,71,76,81,86]
anion_index = list(range(7))
anion_names = ['O3','O2N','ON2','N3','O2F','OFN','O2S']
anion_mendeleev = [261, 256, 251, 246, 267, 262, 262]
anion_mendeleev_rank = [3,2,1,0,6,5,4] # rank based on mendeleev number



#CONVERSION DICTIONARIES
main2atomic = dict(zip(main_index, atomic_index))
atomic2main = dict(zip(atomic_index, main_index))
name2atomic = dict(zip(name_index,atomic_index))
atomic2name = dict(zip(atomic_index,name_index))
main2name = dict(zip(main_index, name_index))
name2main = dict(zip(name_index,main_index))
anion_name2index = dict(zip(anion_names,anion_index))
anion_index2name = dict(zip(anion_index,anion_names))
name2mendeleev = dict(zip(name_index,mendeleev_index))
anion_mendeleev_rank2name = dict(zip(anion_mendeleev_rank, anion_names))
anion_name2mendeleev_rank = dict(zip(anion_names, anion_mendeleev_rank))

mendeleev_rank2name = dict(zip(main_index,sorted(name2mendeleev, key=name2mendeleev.get)))
name2mendeleev_rank = dict(zip(sorted(name2mendeleev, key=name2mendeleev.get), main_index))



# EXCLUSIONS
# TODO: implement chemical rules via this exclusion array(?)
exclusions = []



# FITNESS EVALUATORS AND REQUIREMENTS
def eval_fitness_simple(gap_dir, gap_ind, heat_of_formation, vb_dir, cb_dir, vb_ind, cb_ind):
    stab_score = 0
    gap_dir_score = 0
    gap_ind_score = 0

    if (gap_dir >= 1.5 and gap_dir <= 3):
        gap_dir_score += 10

    if (gap_ind >= 1.5 and gap_ind <= 3):
        gap_ind_score += 10

    if heat_of_formation <= 0.5:
        stab_score += 5

    if heat_of_formation <= 0.2:
        stab_score += 5

    if (vb_dir >= 5.73):
        gap_dir_score += 5

    if (cb_dir <= 4.5):
        gap_dir_score += 5

    if (vb_ind >= 5.73):
        gap_ind_score += 5

    if (cb_ind <= 4.5):
        gap_ind_score += 5

    return max(gap_ind_score, gap_dir_score) + stab_score

def eval_fitness_complex(gap_dir, gap_ind, heat_of_formation, vb_dir, cb_dir, vb_ind, cb_ind):
    stab_score = 0
    gap_dir_score = 0
    gap_ind_score = 0

    if (gap_dir >= 1.5 and gap_dir <= 3):
        gap_dir_score += 10
    elif gap_dir == 0:
        gap_dir_score += 0
    else:
        gap_dir_score += 33 * gaussian_pdf(gap_dir, 2.25)

    if (gap_ind >= 1.5 and gap_ind <= 3):
        gap_ind_score += 10
    elif gap_ind == 0:
        gap_ind_score += 0
    else:
        gap_ind_score += 33 * gaussian_pdf(gap_ind, 2.25)

    if heat_of_formation <= 0.2:
        stab_score = 10
    else:
        stab_score = 20 * (1 - 1 / (1 + math.exp(((-heat_of_formation) + 0.2) * 3.5)))

    if vb_dir >= 5.73:
        gap_dir_score += 5
    else:
        distance = (5.73 - vb_dir) * 5
        gap_dir_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

    if vb_ind >= 5.73:
        gap_ind_score += 5
    else:
        distance = (5.73 - vb_ind) * 5
        gap_ind_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

    if cb_dir <= 4.5:
        gap_dir_score += 5
    else:
        distance = (cb_dir - 4.5) * 5
        gap_dir_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

    if cb_ind <= 4.5:
        gap_ind_score += 5
    else:
        distance = (cb_ind - 4.5) * 5
        gap_ind_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

    return max(gap_ind_score, gap_dir_score) + stab_score

def eval_fitness_complex_product(gap_dir, gap_ind, heat_of_formation, vb_dir, cb_dir, vb_ind, cb_ind):
    stab_score = 0
    gap_dir_score = 0
    gap_ind_score = 0

    if (gap_dir >= 1.5 and gap_dir <= 3):
        gap_dir_score += 10
    elif gap_dir == 0:
        gap_dir_score += 0
    else:
        gap_dir_score += 33 * gaussian_pdf(gap_dir, 2.25)

    if (gap_ind >= 1.5 and gap_ind <= 3):
        gap_ind_score += 10
    elif gap_ind == 0:
        gap_ind_score += 0
    else:
        gap_ind_score += 33 * gaussian_pdf(gap_ind, 2.25)

    if heat_of_formation <= 0.2:
        stab_score = 10
    else:
        stab_score = 20 * (1 - 1 / (1 + math.exp(((-heat_of_formation) + 0.2) * 3.5)))

    if vb_dir >= 5.73:
        gap_dir_score += 5
    else:
        distance = (5.73 - vb_dir) * 5
        gap_dir_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

    if vb_ind >= 5.73:
        gap_ind_score += 5
    else:
        distance = (5.73 - vb_ind) * 5
        gap_ind_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

    if cb_dir <= 4.5:
        gap_dir_score += 5
    else:
        distance = (cb_dir - 4.5) * 5
        gap_dir_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

    if cb_ind <= 4.5:
        gap_ind_score += 5
    else:
        distance = (cb_ind - 4.5) * 5
        gap_ind_score += 10 * (1 - 1 / (1 + math.exp(-distance)))

    return max(gap_ind_score, gap_dir_score) * stab_score * 0.15

def gaussian_pdf(x, mean=0, width=0.5):
    return (1 / math.sqrt(2 * math.pi)) * math.exp(-width * (x - mean) * (x - mean))



# DATA CONVERSION
def raw_tuple_to_data(tuple):
    A = main2name[tuple[0]]
    B = main2name[tuple[1]]
    X = anion_index2name[tuple[2]]
    doc = unc.find({"A":A,"B":B,"anion":X})
    data_doc={}
    for document in doc:
        data_doc = document
    vb_dir = data_doc['VB_dir']
    cb_dir = data_doc['CB_dir']
    vb_ind = data_doc['VB_ind']
    cb_ind = data_doc['CB_ind']
    heat_of_formation = data_doc['heat_of_formation_all']
    gap_dir = abs(cb_dir - vb_dir)
    gap_ind = abs(cb_ind - vb_ind)
    all_the_data = {'vb_dir':vb_dir,'cb_dir':cb_dir,'vb_ind':vb_ind,'cb_ind':cb_ind,
                     'gap_dir':gap_dir,'gap_ind':gap_ind, 'heat_of_formation':heat_of_formation}
    return [all_the_data,document]

def name_to_data(strings):
    A = strings[0]
    B = strings[1]
    X = strings[2]
    doc = unc.find({"A": A, "B": B, "anion": X})
    data_doc = {}
    for document in doc:
        data_doc = document
    vb_dir = data_doc['VB_dir']
    cb_dir = data_doc['CB_dir']
    vb_ind = data_doc['VB_ind']
    cb_ind = data_doc['CB_ind']
    heat_of_formation = data_doc['heat_of_formation_all']
    gap_dir = abs(cb_dir - vb_dir)
    gap_ind = abs(cb_ind - vb_ind)
    all_the_data = {'vb_dir': vb_dir, 'cb_dir': cb_dir, 'vb_ind': vb_ind, 'cb_ind': cb_ind,
                    'gap_dir': gap_dir, 'gap_ind': gap_ind, 'heat_of_formation': heat_of_formation}
    return [all_the_data, document]

def mendeleev_rank_to_data(tuple):
    A = mendeleev_rank2name[tuple[0]]
    B = mendeleev_rank2name[tuple[1]]
    X = anion_mendeleev_rank2name[tuple[2]]
    doc = unc.find({"A": A, "B": B, "anion": X})
    data_doc = {}
    for document in doc:
        data_doc = document
    vb_dir = data_doc['VB_dir']
    cb_dir = data_doc['CB_dir']
    vb_ind = data_doc['VB_ind']
    cb_ind = data_doc['CB_ind']
    heat_of_formation = data_doc['heat_of_formation_all']
    gap_dir = abs(cb_dir - vb_dir)
    gap_ind = abs(cb_ind - vb_ind)
    all_the_data = {'vb_dir': vb_dir, 'cb_dir': cb_dir, 'vb_ind': vb_ind, 'cb_ind': cb_ind,
                    'gap_dir': gap_dir, 'gap_ind': gap_ind, 'heat_of_formation': heat_of_formation}
    return [all_the_data, document]

def mendeleev_mixed_to_data(mixed_tuple):
    A = mendeleev_rank2name[mixed_tuple[0]]
    B = mendeleev_rank2name[mixed_tuple[1]]
    X = mixed_tuple[2]
    doc = unc.find({"A": A, "B": B, "anion": X})
    data_doc = {}
    for document in doc:
        data_doc = document
    vb_dir = data_doc['VB_dir']
    cb_dir = data_doc['CB_dir']
    vb_ind = data_doc['VB_ind']
    cb_ind = data_doc['CB_ind']
    heat_of_formation = data_doc['heat_of_formation_all']
    gap_dir = abs(cb_dir - vb_dir)
    gap_ind = abs(cb_ind - vb_ind)
    all_the_data = {'vb_dir': vb_dir, 'cb_dir': cb_dir, 'vb_ind': vb_ind, 'cb_ind': cb_ind,
                    'gap_dir': gap_dir, 'gap_ind': gap_ind, 'heat_of_formation': heat_of_formation}
    return [all_the_data, document]

def get_input_from_actions(actions, X):
    output = []
    if len(actions) == 1:
        output = X[actions[0]]
    else:
        for action in actions:
            output.append(X[action])
    return output

def get_actions_from_input(input_list, X):
    actions = []
    for input_vector in input_list:
        actions.append(X.index(tuple(input_vector)))
    return actions



# OPTIMIZATION EFFECT GRAPHERS
def atomic_integer_optimization_scatter(iterations=100, guess=(1, 11, 0), fitness_evaluator = eval_fitness_complex):
    #This functions guess argument is in ranked atomic order
    dimensions = [(0, 51), (0, 51), (0, 6)]
    my_output = []
    my_input = []

    # optimizing search
    for i in range(iterations):
        q = raw_tuple_to_data(guess)[0]
        score = -1 * fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                                          q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])
        my_input.append(list(guess))
        my_output.append(score)
        guess = tuple(gp_minimize(my_input, my_output, dimensions))
        print "CALCULATION:", i, " WITH SCORE:", -1 * score

    # finding candidates and recording their existence
    candidate_count = 0
    candidates = []
    for entry in my_input:
        mod_entry = (main2atomic[entry[0]], main2atomic[entry[1]], entry[2])
        print "SEARCHING TO SEE IF ENTRY", mod_entry, "IN GOOD_CANDS_LS"
        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
            candidate_count += 1
            candidates.append(mod_entry)

    print "CANDIDATE COUNT:", candidate_count
    print "CANDIDATES AS ATOMIC INDEX:", candidates

    import matplotlib.pyplot as plt
    plt.plot(list(range(iterations)), [-1 * i for i in my_output], 'g.')
    plt.show()

def categorical_optimization_scatter(iterations=100,guess=("Li","V","O3"), fitness_evaluator=eval_fitness_complex):
    my_input = []
    my_output = []
    dimensions = [name_index, name_index, anion_names]

    guess = list(guess)
    # optimizing search
    for i in range(iterations):
        q = name_to_data(guess)[0]
        score = -1 * fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                                          q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])
        my_input.append(list(guess))
        my_output.append(score)
        guess = gp_minimize(my_input, my_output, dimensions)
        print "CALCULATION:", i, " WITH SCORE:", -1 * score

    # finding candidates and recording their existence
    candidate_count = 0
    candidates = []
    print "SEARCHING TO SEE IF ENTRIES IN GOOD_CANDS_LS"
    for entry in my_input:
        mod_entry = (name2atomic[entry[0]], name2atomic[entry[1]], anion_name2index[entry[2]])
        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
            candidate_count += 1
            candidates.append(mod_entry)

    print "CANDIDATE COUNT:", candidate_count
    print "CANDIDATES AS ATOMIC INDEX:", candidates

    import matplotlib.pyplot as plt
    plt.plot(list(range(iterations)), [-1 * i for i in my_output], 'g.')
    plt.show()

def categorical_optimization_line_and_timing(iterations=100,guess=("Li","V","O3"), fitness_evaluator=eval_fitness_complex):
    import timeit

    my_input = []
    my_output = []
    dimensions = [name_index, name_index, anion_names]

    candidate_count = 0
    candidates = []
    candidate_count_at_iteration=[]
    candidate_iteration = []
    times = []

    # optimizing search
    for i in range(iterations):
        start_time = timeit.default_timer()

        q = name_to_data(guess)[0]
        score = -1 * fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                                       q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])
        my_input.append(list(guess))
        my_output.append(score)

        start_time = timeit.default_timer()
        guess = gp_minimize(my_input, my_output, dimensions)
        elapsed = timeit.default_timer() - start_time
        times.append(elapsed)

        print "CALCULATION:", i+1, " WITH SCORE:", -1 * score

        #Search for entry in GOOD_CANDS_LS
        mod_entry = (name2atomic[my_input[-1][0]], name2atomic[my_input[-1][1]], anion_name2index[my_input[-1][2]])
        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
            candidate_count += 1
            candidates.append(mod_entry)
            candidate_count_at_iteration.append(candidate_count)
            candidate_iteration.append(i)

    print "candidates", candidate_count_at_iteration

    # Plotting
    import matplotlib.pyplot as plt
    candplot = plt.figure(1)
    candline = plt.plot(candidate_iteration, candidate_count_at_iteration)
    plt.setp(candline, linewidth=3, color='g')
    plt.xlabel("Iterations")
    plt.ylabel("Candidates Found")
    plt.title("Candidates vs Iterations")

    timeplot = plt.figure(2)
    timeline = plt.plot(list(range(iterations)), times)
    plt.setp(timeline, linewidth=3, color='b')
    plt.xlabel("Individual Iteration")
    plt.ylabel("Time needed to execute GP")
    plt.title("Computational Overhead of Optimization Algorithm")
    plt.show()

def mendeleev_integer_optimization_line_and_timing(iterations=100, guess=("Li","V","O3"),
                                                   fitness_evaluator=eval_fitness_complex, plots="off"):
    import timeit

    guess = (name2mendeleev_rank[guess[0]],name2mendeleev_rank[guess[1]], anion_name2mendeleev_rank[guess[2]])
    dimensions = [(0, 51), (0, 51), (0, 6)]
    my_output = []
    my_input = []

    candidate_count = 0
    candidates = []
    candidate_count_at_iteration = []
    candidate_iteration = []
    times = []

    # optimizing search
    for i in range(iterations):
        start_time = timeit.default_timer()

        q = mendeleev_rank_to_data(guess)[0]
        score = -1 * fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                                       q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])
        my_input.append(list(guess))
        my_output.append(score)

        start_time = timeit.default_timer()
        guess = gp_minimize(my_input, my_output, dimensions)
        elapsed = timeit.default_timer() - start_time
        times.append(elapsed)

        print "CALCULATION:", i + 1, " WITH SCORE:", -1 * score

        # Search for entry in GOOD_CANDS_LS
        transform_entry = (mendeleev_rank2name[my_input[-1][0]], mendeleev_rank2name[my_input[-1][1]],
                           anion_mendeleev_rank2name[my_input[-1][2]])
        mod_entry = (name2atomic[transform_entry[0]], name2atomic[transform_entry[1]], anion_name2index[transform_entry[2]])
        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
            candidate_count += 1
            candidates.append(mod_entry)
            candidate_count_at_iteration.append(candidate_count)
            candidate_iteration.append(i)

    print "candidates", candidate_count
    print "These candidates are: ", candidates

    # Plotting
    if plots == "on":
        candplot = plt.figure(1)
        candline = plt.plot(candidate_iteration, candidate_count_at_iteration)
        plt.setp(candline, linewidth=3, color='g')
        plt.xlabel("Iterations")
        plt.ylabel("Candidates Found")
        plt.title("Candidates vs Iterations")

        timeplot = plt.figure(2)
        timeline = plt.plot(list(range(iterations)), times)
        plt.setp(timeline, linewidth=3, color='b')
        plt.xlabel("Individual Iteration")
        plt.ylabel("Time needed to execute GP")
        plt.title("Computational Overhead of Optimization Algorithm")
        plt.show()
    return candidate_iteration, candidate_count_at_iteration, list(range(iterations)), times

def mendeleev_mixed_optimization_line_and_timing(iterations=100, guess=("Li", "V", "O3"),
                                                       fitness_evaluator=eval_fitness_complex):

    """
    This functions works in the optimization space of A and B using ranked mendeleev numbers
    and the X space using categories.
    """
    import timeit

    guess = (name2mendeleev_rank[guess[0]], name2mendeleev_rank[guess[1]], guess[2])
    dimensions = [(0, 51), (0, 51), anion_names]
    my_output = []
    my_input = []

    candidate_count = 0
    candidates = []
    candidate_count_at_iteration = []
    candidate_iteration = []
    times = []

    # optimizing search
    for i in range(iterations):
        start_time = timeit.default_timer()

        q = mendeleev_mixed_to_data(guess)[0]
        score = -1 * fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                                       q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])
        my_input.append(list(guess))
        my_output.append(score)

        start_time = timeit.default_timer()
        guess = gp_minimize(my_input, my_output, dimensions)
        elapsed = timeit.default_timer() - start_time
        times.append(elapsed)

        print "CALCULATION:", i + 1, " WITH SCORE:", -1 * score

        # Search for entry in GOOD_CANDS_LS
        transform_entry = (
        mendeleev_rank2name[my_input[-1][0]], mendeleev_rank2name[my_input[-1][1]], my_input[-1][2])
        mod_entry = (name2atomic[transform_entry[0]], name2atomic[transform_entry[1]], anion_name2index[transform_entry[2]])

        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
            candidate_count += 1
            candidates.append(mod_entry)
            candidate_count_at_iteration.append(candidate_count)
            candidate_iteration.append(i)

    print candidate_count, "candidates were found at iterations", candidate_iteration
    print "These candidates are: ", candidates

    # Plotting

    candplot = plt.figure(1)
    candline = plt.plot(candidate_iteration, candidate_count_at_iteration)
    plt.setp(candline, linewidth=3, color='g')
    plt.xlabel("Iterations")
    plt.ylabel("Candidates Found")
    plt.title("Candidates vs Iterations")

    timeplot = plt.figure(2)
    timeline = plt.plot(list(range(iterations)), times)
    plt.setp(timeline, linewidth=3, color='b')
    plt.xlabel("Individual Iteration")
    plt.ylabel("Time needed to execute GP")
    plt.title("Computational Overhead of Optimization Algorithm")
    plt.show()

def mendeleev_integer_optimization_combo_line_and_timing(iterations=100, guess=("Li", "V", "O3"),
                                                         fitness_evaluator=eval_fitness_complex, plots="off"):
    import timeit

    guess = (name2mendeleev_rank[guess[0]],name2mendeleev_rank[guess[1]], anion_name2mendeleev_rank[guess[2]])
    dimensions = [(0, 51), (0, 51), (0, 6)]
    my_output = []
    my_input = []

    candidate_count = 0
    candidates = []
    candidate_count_at_iteration = []
    candidate_iteration = []
    times = []

    X = calculate_discrete_space(dimensions)

    # optimizing search
    for i in range(iterations):
        start_time = timeit.default_timer()

        q = mendeleev_rank_to_data(guess)[0]
        score = fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                                       q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])
        my_input.append(list(guess))
        my_output.append(score)

        start_time = timeit.default_timer()

        prev_actions = get_actions_from_input(my_input, X)
        policy = combo.search.discrete.policy(test_X=np.asarray(X))
        policy.write(prev_actions, np.asarray(my_output))
        actions = policy.bayes_search(max_num_probes=1, num_search_each_probe=1,
                                      simulator=None, score='EI', interval=0, num_rand_basis=0)
        guess = list(get_input_from_actions(actions, X))

        elapsed = timeit.default_timer() - start_time
        times.append(elapsed)

        print "CALCULATION:", i + 1, " WITH SCORE:", score

        # Search for entry in GOOD_CANDS_LS
        transform_entry = (mendeleev_rank2name[my_input[-1][0]], mendeleev_rank2name[my_input[-1][1]],
                           anion_mendeleev_rank2name[my_input[-1][2]])
        mod_entry = (name2atomic[transform_entry[0]], name2atomic[transform_entry[1]], anion_name2index[transform_entry[2]])
        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
            candidate_count += 1
            candidates.append(mod_entry)
            candidate_count_at_iteration.append(candidate_count)
            candidate_iteration.append(i)

    print "candidates", candidate_count
    print "These candidates are: ", candidates

    # Plotting
    if plots =="on":
        candplot = plt.figure(1)
        candline = plt.plot(candidate_iteration, candidate_count_at_iteration)
        plt.setp(candline, linewidth=3, color='g')
        plt.xlabel("Iterations")
        plt.ylabel("Candidates Found")
        plt.title("Candidates vs Iterations")

        timeplot = plt.figure(2)
        timeline = plt.plot(list(range(iterations)), times)
        plt.setp(timeline, linewidth=3, color='b')
        plt.xlabel("Individual Iteration")
        plt.ylabel("Time needed to execute GP")
        plt.title("Computational Overhead of Optimization Algorithm")
        plt.show()

    return candidate_iteration, candidate_count_at_iteration, list(range(iterations)), times

def mendeleev_integer_statistical_comparisons(iter_num=5, run_num=5, initial_guessing="random"):
    '''Run parameters'''

    combo_cands = []
    skopt_cands = []
    combo_iters = []
    skopt_iters = []
    combo_times = []
    skopt_times = []
    iterations = []

    '''Running computations'''
    for i in range(run_num):
        if initial_guessing=="random":
            initial_guess = dummy_minimize([name_index, name_index, anion_names])
        else:
            initial_guess = initial_guessing

        skopt_cand_iter, skopt_cand_count_at_iter, iterations, skopt_time = \
            mendeleev_integer_optimization_line_and_timing( guess=initial_guess, iterations=iter_num)
        combo_cand_iter, combo_cand_count_at_iter, iterations, combo_time = \
            mendeleev_integer_optimization_combo_line_and_timing(guess=initial_guess, iterations=iter_num)

        skopt_times.append(skopt_time)
        combo_times.append(combo_time)

        skopt_cands.append(skopt_cand_count_at_iter)
        combo_cands.append(combo_cand_count_at_iter)

        skopt_iters.append(skopt_cand_iter)
        combo_iters.append(combo_cand_iter)

    print "finished optimizing runs"

    '''Save Results'''
    text_file = open('results_raw.txt', 'w')
    text_file.write("\n TIME OF RUN: {} \n".format(datetime.datetime.now().time().isoformat()))
    text_file.write("skopt raw iterations: {} \n".format(skopt_iters))
    text_file.write("skopt raw candidate list: {} \n".format(skopt_cands))
    text_file.write("combo raw iterations: {} \n".format(combo_iters))
    text_file.write("combo raw candidate list: {} \n".format(combo_cands))

    '''Data Reformatting'''
    def get_time_stats(gp_times):
        avg_times = []
        for i in range(len(gp_times[0])):
            temp = []
            for time_arr in gp_times:
                temp.append(time_arr[i])
            avg_times.append(np.asarray(temp).mean())
        return avg_times

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


    combo_times = get_time_stats(combo_times)
    skopt_times = get_time_stats(skopt_times)


    skopt_iters, skopt_iters_std, skopt_cands = get_cand_stats(skopt_cands, skopt_iters)
    combo_iters, combo_iters_std, combo_cands = get_cand_stats(combo_cands, combo_iters)

    '''Save Results'''
    text_file = open('results_processed.txt', 'w')
    text_file.write("\n TIME OF RUN: {} \n".format(datetime.datetime.now().time().isoformat()))
    text_file.write("skopt iterations: {} \n".format(skopt_iters))
    text_file.write("skopt std dev iterations: {} \n".format(skopt_iters_std))
    text_file.write("skopt candidate list: {} \n".format(skopt_cands))
    text_file.write("combo iterations: {} \n".format(combo_iters))
    text_file.write("combo std dev iterations: {} \n".format(combo_iters_std))
    text_file.write("combo candidate list: {} \n".format(combo_cands))

    '''Plotting'''
    candplot = plt.figure(1)
    skopterr = plt.errorbar(skopt_iters, skopt_cands, xerr=skopt_iters_std, fmt='og', ecolor='black',
                            capthick=2, capsize=3, elinewidth=2)
    skoptline = plt.plot(skopt_iters, skopt_cands, 'g')

    comboerr = plt.errorbar(combo_iters, combo_cands, xerr=combo_iters_std, fmt='ob', ecolor='black',
                            capthick=2, capsize=3, elinewidth=2)

    comboline = plt.plot(combo_iters, combo_cands, 'b')

    rand_iters = []
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

    timeplot = plt.figure(2)
    skopt_timeline = plt.plot(iterations, skopt_times)
    combo_timeline = plt.plot(iterations, combo_times)
    plt.setp(skopt_timeline, linewidth=3, color='g')
    plt.setp(combo_timeline, linewidth=3, color='b')
    plt.xlabel("Individual Iteration")
    plt.ylabel("Time needed to execute GP")
    plt.title("Computational Overhead of Optimization Algorithm")
    plt.show()

# EXECUTABLE
if __name__ =="__main__":
    # mendeleev_integer_optimization_combo_line_and_timing(iterations=1000, guess = ['Os','Os','O3'])

    # informed comparison
    mendeleev_integer_statistical_comparisons(iter_num=500, run_num= 20, initial_guessing="random")











