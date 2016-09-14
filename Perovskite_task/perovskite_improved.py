from perovskite_task import *
from sigopt import Connection
import time

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
def graph_one(candidate_iterations, candidate_count, text):
    candplot = plt.figure(1)
    candline = plt.plot(candidate_iterations, candidate_count)
    plt.setp(candline, linewidth=3, color='g')
    plt.xlabel("Iterations")
    plt.ylabel("Candidates Found")
    plt.title("{}: Candidates vs Iterations".format(text))
    plt.show()
def graph_all():
    pass

# ASSORTED FUNCTIONS
def del_sigopt_connection(experiment_id, token="VEXZNUGYFSANVIWWDECDVMUELNFIAVWNJXQDSXALTMBOKAJQ"):
    conn = Connection(client_token=token)
    conn.experiments(experiment_id).delete()

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
    candidate_count_at_iteration = [0]
    candidate_iteration = [0]
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

    if exclusions==None:
        exclusions=[]

    guess = (name2mendeleev_rank[guess[0]],name2mendeleev_rank[guess[1]], anion_name2mendeleev_rank[guess[2]])
    dimensions = [(0, 51), (0, 51), (0, 6)]
    my_output = []
    my_input = []

    candidate_count = 0
    candidates = []
    candidate_count_at_iteration = [0]
    candidate_iteration = [0]
    times = []
    current_iteration=0

    # optimizing search
    while candidate_count != cands:
        current_iteration+=1
        X = calculate_discrete_space(dimensions)
        for exclusion in exclusions:
            X.remove(exclusion)
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
def sigopt_trial(cands=1,guess=("Li","V","O3"),fitness_evaluator=eval_fitness_complex,exclusions=None,
                 token="VEXZNUGYFSANVIWWDECDVMUELNFIAVWNJXQDSXALTMBOKAJQ", experiment_id=None):

    #todo: implement duplicate check

    conn = Connection(client_token=token)

    if exclusions==None:
        exclusions=[]

    guess = (name2mendeleev_rank[guess[0]],name2mendeleev_rank[guess[1]], anion_name2mendeleev_rank[guess[2]])

    candidate_count = 0
    candidates = []
    candidate_count_at_iteration = [0]
    candidate_iteration = [0]
    times = []
    current_iteration=0
    true_iteration=0

    if experiment_id==None:
        experiment = conn.experiments().create(
            name='SigOpt Perovskite Trial',
            parameters=[
                dict(name='A', type='int', bounds=dict(min=0, max=51)),
                dict(name='B', type='int', bounds=dict(min=0, max=51)),
                dict(name='anion', type='int', bounds=dict(min=0, max=6))],)
    else:
        class ExperimentClass():
            id = experiment_id
        experiment=ExperimentClass()

    print "experiment ID", experiment.id

    def evaluate_model(assignments):
        assign_A = assignments['A']
        assign_B = assignments['B']
        assign_X = assignments['anion']
        q = mendeleev_rank_to_data((assign_A, assign_B, assign_X))[0]
        fit_score = fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                                  q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])
        return fit_score

    while candidate_count != cands:
        current_iteration+=1
        true_iteration+=1
        start_time = time.time()

        if true_iteration > 1200: # avoiding overrunning sigopt limit
            break

        # Have SigOpt create suggestion and score data
        guess = conn.experiments(experiment.id).suggestions().create()
        tuple_guess = (guess.assignments['A'], guess.assignments['B'], guess.assignments['anion'])
        if tuple_guess in exclusions:
            current_iteration-=1
            score = 0
        else:
            score = evaluate_model(guess.assignments)
            print "SIGOPT CALCULATION", current_iteration, "WITH SCORE", score

        observation = conn.experiments(experiment.id).observations().create(
            suggestion=guess.id,
            value=score,
        )

        # uncomment this to go back to deleting the observations, problem is that sigopt runs same stuff over again
        # if tuple_guess in exclusions:
        #     conn.experiments(experiment.id).observations(observation.id).delete()
        #     continue

        elapsed = time.time() - start_time
        times.append(elapsed)

        # Search for entry in GOOD_CANDS_LS
        transform_entry = (mendeleev_rank2name[tuple_guess[0]], mendeleev_rank2name[tuple_guess[1]],
                           anion_mendeleev_rank2name[tuple_guess[2]])
        mod_entry = (
        name2atomic[transform_entry[0]], name2atomic[transform_entry[1]], anion_name2index[transform_entry[2]])
        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
            candidate_count += 1
            candidates.append(mod_entry)
            candidate_count_at_iteration.append(candidate_count)
            candidate_iteration.append(current_iteration)

    print "SIGOPT CANDIDATES", candidates
    return candidate_iteration, candidate_count_at_iteration, times
def chemical_rules_trial(cands=1,ranking=gs_rank, exclusions=chemical_exclusions):
    candidate_count=0
    candidates=[]
    candidate_count_at_iteration=[0]
    candidate_iteration=[0]
    current_iteration = 0
    if exclusions==None:
        exclusions=[]

    for entry in ranking:
        if cands == candidate_count:
            break
        elif entry not in exclusions:
            current_iteration+=1
            if entry in GOOD_CANDS_LS and entry not in candidates:
                candidate_count += 1
                candidates.append(entry)
                candidate_count_at_iteration.append(candidate_count)
                candidate_iteration.append(current_iteration)
                print "CRS CALCULATION: ", current_iteration, "WITH SCORE: 30"
            else:
                print "CRS CALCULATION: ", current_iteration, "WITH SCORE: 0"

        else: # it is an exclusion and the candidates have not met candidate count
            print "CRS CALCULATION: ", current_iteration, "WITH SCORE: 0"
            continue

    print "CHEM RULE CANDIDATES", candidates
    return candidate_iteration, candidate_count_at_iteration
def random_trial(cands=1):
    candidate_count_at_iteration = list(range(cands))
    candidate_iteration = [946.4 * element for element in candidate_count_at_iteration]
    return candidate_iteration, candidate_count_at_iteration

# REPEATING/PARALLEL TRIALS
def multiprocessor(trial_types, trial_runs):
    '''
    skopt_trial => 'skopt'
    combo_trial => 'combo'
    chemical_rules_trial => 'cherules'
    random_trial => 'random'
    sigopt_trial => 'sigopt'
    :param trial_types: ['sigopt', 'random', 'skopt']
    :param trial_runsl: [20, 1, 20]
    '''
    def worker():
        pass
    pass

# GRAPHING STUFF
def rules_vs_rules_and_exclusions_gs_rank():
    crs_iteration, crs_count_at_iteration = chemical_rules_trial(cands=20, exclusions=chemical_exclusions)
    crs_iteration2, crs_count_at_iteration2 = chemical_rules_trial(cands=20, exclusions=None)
    rs_iteration, rs_count_at_iteration = random_trial(cands=20)

    candplot = plt.figure(1)
    candline = plt.plot(crs_iteration,  crs_count_at_iteration)
    candline2 = plt.plot(crs_iteration2, crs_count_at_iteration2)
    candline3 = plt.plot(rs_iteration, rs_count_at_iteration)
    plt.setp(candline, linewidth=3, color='g')
    plt.setp(candline2, linewidth=3, color='r')
    plt.setp(candline3, linewidth=3, color='black')
    plt.xlabel("Iterations")
    plt.ylabel("Candidates Found")
    plt.title("{}: Candidates vs Iterations".format("Chemical Rules Search"))
    plt.show()

if __name__== "__main__":

    # del_sigopt_connection(experiment_id=7403)
    # so_iters, so_cands, so_times = sigopt_trial(cands=1, exclusions=mendeleev_chemical_exclusions)
    s_iters, s_cands, s_times = skopt_trial(cands=1, exclusions=mendeleev_chemical_exclusions)

    # save_it(so_iters, so_cands,name="sigopt")
    print s_iters, s_cands
    # print so_iters, so_cands


    # c_iteration, c_count_at_iteration, c_times = combo_trial(cands=5, guess=("Li","Al","N3"))
    # c_ex_iteration, c_ex_count_at_iteration, c_ex_times = combo_trial(cands=5, guess=("Li","Al","N3"),
    #                                                                   exclusions=mendeleev_chemical_exclusions)

