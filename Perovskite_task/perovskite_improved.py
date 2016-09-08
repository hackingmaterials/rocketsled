from perovskite_task import *

# SAVING AND GRAPHING
def save_it(trial_iters, trial_cands, trial_times=None, name="placeholder"):

    '''Save Results'''
    with open('results.txt', 'a') as text_file:
        day = str(datetime.date.today())
        time = datetime.datetime.now().time().isoformat()
        text_file.write("\n\nRUN COMPLETED: {} - {}".format(day,time))
        text_file.write("\n    RAW DATA\n")
        text_file.write("    {} raw iterations: {} \n".format(name, trial_iters))
        text_file.write("    {} raw candidate list: {} \n".format(name, trial_cands))

    # Data reformatting
    if trial_times==None:
        pass
    else:
        trial_times = get_time_stats(trial_times)
        pickle.dump(trial_times, open("{}_times.p".format(name), "wb"))

    trial_iters, trial_iters_std, trial_cands = get_cand_stats(trial_cands, trial_iters)

    # Save reformatted results
    with open('results.txt', 'a') as text_file:
        text_file.write("\n    PROCESSED DATA \n")
        text_file.write("    {} iterations: {} \n".format(name,trial_iters))
        text_file.write("    {} std dev iterations: {} \n".format(name, trial_iters_std))
        text_file.write("    {} candidate list: {} \n".format(name, trial_cands))
def graph_one():
    pass
def graph_all():
    pass

# INIDIVIDUAL TRIALS
def skopt_trial(cands=1, guess=("Li", "V", "O3"),fitness_evaluator=eval_fitness_complex,exclusions=None):
    '''A complete, individual skopt trial
    Remeber, the exclusions should be in mendeleev form'''

    if exclusions == None:
        exclusions = []
    guess = (name2mendeleev_rank[guess[0]], name2mendeleev_rank[guess[1]], anion_name2mendeleev_rank[guess[2]])
    dimensions = [(0, 51), (0, 51), (0, 6)]
    my_output = []
    my_input = []

    candidate_count = 0
    candidates = []
    candidate_count_at_iteration = []
    candidate_iteration = []
    times = []
    current_iteration = 0

    # optimizing search
    while candidate_count != cands:
        current_iteration+=1
        start_time = timeit.default_timer()

        continue_exclusions=True
        while continue_exclusions:
            q = mendeleev_rank_to_data(guess)[0]
            score = -1 * fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                                           q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])

            # To prevent a good first guess from being overwritten
            if current_iteration==1:
                if tuple(guess) not in exclusions:
                    my_input.append(list(guess))
                    my_output.append(score)
                    break

            my_input.append(list(guess))
            my_output.append(score)
            guess = gp_minimize(my_input, my_output, dimensions)

            if tuple(guess) not in exclusions:
                continue_exclusions=False
            elif tuple(guess) in exclusions:
                # print "MENDELEEV RANK:", tuple(guess), "EXCLUDED!"
                my_output[-1] = 0
        # print "my_input, my_output

        elapsed = timeit.default_timer() - start_time
        times.append(elapsed)

        print "SKOPT CALCULATION:", current_iteration  , " WITH SCORE:", -1 * score

        # Search for entry in GOOD_CANDS_LS
        transform_entry = (mendeleev_rank2name[my_input[-1][0]], mendeleev_rank2name[my_input[-1][1]],
                           anion_mendeleev_rank2name[my_input[-1][2]])
        mod_entry = (name2atomic[transform_entry[0]], name2atomic[transform_entry[1]],
                     anion_name2index[transform_entry[2]])
        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
            candidate_count += 1
            candidates.append(mod_entry)
            candidate_count_at_iteration.append(candidate_count)
            candidate_iteration.append(current_iteration)

    print "SKOPT CANDIDATES", candidates

    return candidate_iteration, candidate_count_at_iteration, times
def combo_trial(cands=1, guess=("Li", "V", "O3"),fitness_evaluator=eval_fitness_complex,exclusions=None):
    '''A complete, individual combo trial'''
    @contextmanager
    def suppress_stdout():
        with open(os.devnull, "w") as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    guess = (name2mendeleev_rank[guess[0]],name2mendeleev_rank[guess[1]], anion_name2mendeleev_rank[guess[2]])
    dimensions = [(0, 51), (0, 51), (0, 6)]
    my_output = []
    my_input = []

    candidate_count = 0
    candidates = []
    candidate_count_at_iteration = []
    candidate_iteration = []
    times = []
    current_iteration=0

    # optimizing search
    while candidate_count != cands:
        current_iteration+=1
        X = calculate_discrete_space(dimensions)
        print "PREV LEN:", len(X)
        for exclusion in exclusions:
            X.remove(exclusion)
        print "LEN:", len(X)
        print X[3234]
        start_time = timeit.default_timer()

        q = mendeleev_rank_to_data(guess)[0]
        score = fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                                       q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])
        my_input.append(list(guess))
        my_output.append(score)

        start_time = timeit.default_timer()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with suppress_stdout():       #basically supressing all combo spam and scikit-learn spams

                prev_actions = get_actions_from_input(my_input, X)
                policy = combo.search.discrete.policy(test_X=np.asarray(X))
                policy.write(prev_actions, np.asarray(my_output))
                actions = policy.bayes_search(max_num_probes=1, num_search_each_probe=1,
                                              simulator=None, score='EI', interval=0, num_rand_basis=0)
                primary_guess = list(get_input_from_actions(actions, X))

                guess = duplicate_check(primary_guess, my_input, X, "combo")

                elapsed = timeit.default_timer() - start_time
                times.append(elapsed)

        print "COMBO CALCULATION:", current_iteration , " WITH SCORE:", score

        # Search for entry in GOOD_CANDS_LS
        transform_entry = (mendeleev_rank2name[my_input[-1][0]], mendeleev_rank2name[my_input[-1][1]],
                           anion_mendeleev_rank2name[my_input[-1][2]])
        mod_entry = (name2atomic[transform_entry[0]], name2atomic[transform_entry[1]], anion_name2index[transform_entry[2]])
        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
            candidate_count += 1
            candidates.append(mod_entry)
            candidate_count_at_iteration.append(candidate_count)
            candidate_iteration.append(current_iteration)

    print "COMBO CANDIDATES", candidates

    return candidate_iteration, candidate_count_at_iteration, times
def chemical_rules_trial(cands=1):
    pass
def random_trial(cands=1):
    pass
def sigopt_trial(cands=1):
    pass

# REPEATING/PARALLEL TRIALS
def multiprocessor():
    pass


if __name__== "__main__":
    # print len(exclusions)
    # combo_trial(cands=2)
    # combo_trial(cands=2, exclusions=mendeleev_chemical_exclusions)
    skopt_trial(cands=2, exclusions=mendeleev_chemical_exclusions)
    # skopt_trial(cands=2)
    # if (0, 14, 3) in mendeleev_chemical_exclusions:
    #     print "yep"
    # else:
    #     print "nope"



