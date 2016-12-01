from perovskite_task import *
from sigopt import Connection
from sigopt.exception import ConnectionException, ApiException
import time
import signal
import csv

dev = "BRSOTAQXVWFBVRLFXLEJYIFXOPKXYXQAUUAXFECVHKJPDPRN"
sig = "VEXZNUGYFSANVIWWDECDVMUELNFIAVWNJXQDSXALTMBOKAJQ"

# SAVING AND GRAPHING
def save_it(trial_iters, trial_cands, name="placeholder", filename = "results.txt", extra_info="none"):

    '''Save Results'''
    with open(filename, 'a') as text_file:
        day = str(datetime.date.today())
        time = datetime.datetime.now().time().isoformat()
        text_file.write("\n\nRUN COMPLETED: {} - {}".format(day,time))
        text_file.write("\n    RAW DATA\n")
        text_file.write("    {} raw iterations: {} \n".format(name, trial_iters))
        text_file.write("    {} raw candidate list: {} \n".format(name, trial_cands))
        text_file.write("    extra info: {} \n".format(extra_info))
def write_it(what, where='observations.txt'):
    # For use with tracking sigopt trials
    with open(where, 'a') as csvfile:
        fieldnames = ['A', 'B', 'anion']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        assignments = {}
        assignments['A'] = what[0]
        assignments['B'] = what[1]
        assignments['X'] = what[2]
        writer.writerow(assignments)




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

# ASSORTED SIGOPT FUNCTIONS
def list_sigopt_experiments(token=sig):
    print "fetching..."
    conn = Connection(client_token=token)
    exps = conn.experiments().fetch().data
    ids = []
    for exp in exps:
        ids.append(int(exp.id))
    print ids
    return ids
def archive_sigopt_experiments(ids, token=sig):
    # careful now
    print "archiving..."
    conn = Connection(client_token=token)
    if type(ids)!=list:
        conn.experiments(ids).delete()
        print "archived", ids, "."
    else:
        for id in ids:
            conn.experiments(id).delete()
            print "archived", id, "."
    print "finding remaining active experiments:"
    return list_sigopt_experiments(token=token)
def list_sigopt_observations(experiment_id, token=sig):
    #returns number of observations for given experiment
    # print "fetching..."
    conn = Connection(client_token=token)
    observations_list = conn.experiments(experiment_id).fetch().progress.observation_count
    return observations_list
def repurpose_sigopt_experiment(experiment_id, token=sig):
    print "repurposing..."
    conn = Connection(client_token=token)
    observation_ids = conn.experiments(experiment_id).observations().delete()
    print "repurposed experiment", experiment_id, "."
def evaluate_model(assignments, fitness_evaluator):
    assign_A = assignments['A']
    assign_B = assignments['B']
    assign_X = assignments['anion']
    guess = (assign_A, assign_B, assign_X)
    # guess = eneg_rank2namefn(guess)
    # q = name_to_data(guess)[0]
    q = mendeleev_rank_to_data((assign_A, assign_B, assign_X))[0]
    fit_score = fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                              q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])
    return fit_score



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
                           mendeleev_rank2anion_name[my_input[-1][2]])
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
                           mendeleev_rank2anion_name[my_input[-1][2]])
        mod_entry = (name2atomic[transform_entry[0]], name2atomic[transform_entry[1]], anion_name2index[transform_entry[2]])
        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
            candidate_count += 1
            candidates.append(mod_entry)
            candidate_count_at_iteration.append(candidate_count)
            candidate_iteration.append(current_iteration)

    print "COMBO CANDIDATES", candidates

    return candidate_iteration, candidate_count_at_iteration, times
def sigopt_trial(cands=1, fitness_evaluator=eval_fitness_complex, experiment_name = "Perovskite trial",
                 exclusions=None, token=sig, experiment_id=None, volcano=False, exhaustive=False, radius=1,
                 dimensions = [(0,51),(0,51),(0,6)], trial_num="None", tracking=False, anion=None):

    conn = Connection(client_token=token)
    timeout = 1200
    save_interval = 500
    upper_limit = 4000

# EXCLUSIONS ASSIGNMENT
    if exclusions==None:
        exclusions=[]
        exclusion_name= "none"
    elif exclusions==mendeleev_chemical_exclusions:
        exclusion_name = "chemical exlcusions (transformed to mendeleev form)"

# VALUES FOR KEEPING STATS ON RUNS
    candidate_count = 0
    candidates = []
    candidate_count_at_iteration = [0]
    candidate_iteration = [0]
    current_iteration=0
    errors = ""
    true_iteration=0
    blacklist = []
    multiplier = 1.5

# CREATE NEW EXPERIMENT OR RERUN
    if experiment_id==None:
        if anion==None:
            experiment = conn.experiments().create(
                name=experiment_name,
                parameters=[
                    dict(name='A', type='int', bounds=dict(min=dimensions[0][0], max=dimensions[0][1])),
                    dict(name='B', type='int', bounds=dict(min=dimensions[1][0], max=dimensions[1][1])),
                    dict(name='anion', type='int', bounds=dict(min=dimensions[2][0], max=dimensions[2][1]))],)
        else:
            experiment = conn.experiments().create(
                name=experiment_name,
                parameters=[
                    dict(name='A', type='int', bounds=dict(min=dimensions[0][0], max=dimensions[0][1])),
                    dict(name='B', type='int', bounds=dict(min=dimensions[1][0], max=dimensions[1][1]))],)
    else:
        class ExperimentClass():
            id = experiment_id
        experiment=ExperimentClass()

    print "experiment ID", experiment.id

    while candidate_count != cands and true_iteration<upper_limit:
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

            fail_state = False
            current_iteration += 1
            true_iteration = list_sigopt_observations((experiment.id), token=token)

            # CREATE SIGOPT SUGGESTION
            guess=None
            tuple_guess=None
            guess = conn.experiments(experiment.id).suggestions().create()
            if anion==None:
                tuple_guess = (guess.assignments['A'], guess.assignments['B'], guess.assignments['anion'])
            else:
                tuple_guess = (guess.assignments['A'], guess.assignments['B'], anion)
            if tracking:
                write_it(tuple_guess,"mendeleev_tracker{}.csv".format(trial_num))
            if tuple_guess in exclusions:
                current_iteration-=1
                score = None
                fail_state = True
            else:
                if anion==None:
                    score = evaluate_model(guess.assignments, fitness_evaluator)
                else:
                    assignments = {'A': guess.assignments['A'], 'B': guess.assignments['B'], 'anion':anion }
                    score = evaluate_model(assignments, fitness_evaluator)


            time.sleep(runsleep)

            if score==30:


                # CHECK CANDIDACY
                mod_entry = mendeleev2atomic(tuple_guess)
                if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
                    print "SCORE OF 30! THE MENDELEEV COMPOUND IS", tuple_guess
                    candidate_count += 1
                    candidates.append(mod_entry)
                    candidate_count_at_iteration.append(candidate_count)
                    candidate_iteration.append(current_iteration)

                if volcano:
                    fail_state=True
                    score=None

            # NUKE MOUNTAIN AROUND 30 RESPONSE
                if exhaustive:
                    # WARNING: THIS IS BROKEN
                    # Beware, this method does not cascade exhaustively.
                    # So if one candidate found, will only exhaustive search once even if finds others in search
                    A_ion = tuple_guess[0]
                    B_ion = tuple_guess[1]
                    X_ion = tuple_guess[2]

                    if A_ion - radius < 0:
                        A_ion_lower = 0
                    else:
                        A_ion_lower = A_ion - radius
                    if A_ion + radius > 52:
                        A_ion_higher = 52
                    else:
                        A_ion_higher = A_ion + radius

                    if B_ion - radius < 0:
                        B_ion_lower = 0
                    else:
                        B_ion_lower = B_ion - radius
                    if B_ion + radius > 52:
                        B_ion_higher = 52
                    else:
                        B_ion_higher = B_ion + radius

                    if X_ion - radius < 0:
                        X_ion_lower = 0
                    else:
                        X_ion_lower = X_ion - radius
                    if X_ion + radius > 7:
                        X_ion_higher = 7
                    else:
                        X_ion_higher = X_ion + radius

                    mountain_dim = [(A_ion_lower, A_ion_higher), (B_ion_lower, B_ion_higher), (X_ion_lower, X_ion_higher)]
                    mountain = calculate_discrete_space(mountain_dim)

                    blacklist += mountain
                    print "EXHAUSTIVE SEARCH WITH NUM CANDS:", len(mountain)

                    # Remove all guessed guesses from mountain
                    for observation in conn.experiments(experiment.id).observations().fetch().iterate_pages():
                        mountain_guess = (observation.assignments["A"], observation.assignments["B"], observation.assignments["anion"])
                        if mountain_guess in mountain:
                            mountain.remove(mountain_guess)
                        if mountain_guess not in blacklist:
                            old_score = observation.value
                            observation = conn.experiments(experiment.id).observations(observation.id).update(
                                assignments=assignments,
                                value=old_score*multiplier)

                    # Record all guesses in mountain
                    for mountain_tuple in mountain:
                        assignments = {'A': mountain_tuple[0], 'B': mountain_tuple[1], 'anion': mountain_tuple[2]}
                        score = evaluate_model(assignments, fitness_evaluator)
                        current_iteration+=1
                        print "SIGOPT CALCULATION", current_iteration, "WITH SCORE:", score, "FAILED STATE:", fail_state
                        observation = conn.experiments(experiment.id).observations().create(
                            assignments=assignments,
                            value=score)
                        if tracking:
                            write_it(mountain_guess, "mendeleev_tracker{}.csv".format(trial_num))

                        mod_entry = mendeleev2atomic(mountain_tuple)
                        if mod_entry in GOOD_CANDS_LS and mod_entry not in candidates:
                            print "SCORE OF 30! THE MENDELEEV COMPOUND IS", mountain_tuple
                            candidate_count += 1
                            candidates.append(mod_entry)
                            candidate_count_at_iteration.append(candidate_count)
                            candidate_iteration.append(current_iteration)

            observation = conn.experiments(experiment.id).observations().create(
                suggestion=guess.id,
                value=score,
                failed = fail_state)



            print "SIGOPT CALCULATION", current_iteration, "WITH SCORE:", score, "FAILED STATE:", fail_state

        # except ConnectionException, E:
        #     errors = (E)
        #     print "HANDLING CONNECTION EXCEPTION:", E
        #     current_iteration-=1
        #     continue
        #
        # except ApiException, APIE:
        #     errors = (APIE)
        #     print "HANDLING API EXCEPTION:", APIE
        #     break

        except ConnectionException, E:
            print "HANDLING CONNECTION EXCEPTION"
            time.sleep(runsleep)
            current_iteration-=1
            continue

        except ApiException, A:
            print "HANDLING API EXCEPTION {}".format(A)
            break

        except RuntimeError, R:
            print "HANDLING OTHER ERROR {}".format(R)
            signal.alarm(timeout)
            continue

        if true_iteration!=0:
            if true_iteration % save_interval == 0:
                save_it(candidate_iteration, candidate_count_at_iteration,
                        filename="temp_results{}{}.txt".format(trial_num, true_iteration))

    info = {'recorded_iterations':current_iteration, "errors":errors,
            "experiment id":experiment.id, 'exclusions': exclusion_name, 'candidates': candidates}

    save_it(candidate_iteration, candidate_count_at_iteration, name="{} iter".format(upper_limit),
            filename="results{}.{}.txt".format(trial_num, true_iteration), extra_info=info)
    print "SIGOPT CANDIDATES", candidates
    return candidate_iteration, candidate_count_at_iteration, info, experiment_id
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
def full_trial(fitness_evaluator=eval_fitness_complex):
    # actually accesses and scores all 18,928 candidates


    filename = 'materials_data.csv'

    with open(filename, 'a') as csvfile:
        fieldnames= ['A', 'B', 'anion', 'complex score', 'simple score', 'complex product']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for A in range(52):
            for B in range(52):
                for X in range(7):
                    assignments = {"A":A, "B":B, "anion":X}
                    score_complex = evaluate_model(assignments, eval_fitness_complex)
                    score_simple = evaluate_model(assignments, eval_fitness_simple)
                    score_product = evaluate_model(assignments, eval_fitness_complex_product)
                    assignments['complex score'] = score_complex
                    assignments['simple score'] = score_simple
                    assignments['complex product'] = score_product
                    writer.writerow(assignments)


                    if score_complex==30 or score_simple==30 or score_product==30:
                        print "cs", score_complex, "ss", score_simple, "sp", score_product
                        # print "mendeleev:", (A,B,X), "atomic", mendeleev2atomic((A, B, X))
                        print "eneg", (A,B,X), "name", eneg_rank2namefn((A,B,X))


# GRAPHING STUFF
def rules_vs_rules_and_exclusions_gs_rank():
    crs_iteration, crs_count_at_iteration = chemical_rules_trial(cands=20, exclusions=chemical_exclusions)
    # rs_iteration, rs_count_at_iteration = random_trial(cands=4)
    print crs_iteration
    print crs_count_at_iteration

    iters = [[0, 117, 194, 448, 618], [0, 176, 198], [0, 328, 425, 929], [0, 125, 132, 133]]
    cands = [[0, 1, 2, 3, 4], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3]]

    # avg_iterations_at_candidate, std_iterations_at_candidate, all_cands=get_cand_stats(iters, cands)
    #
    # print avg_iterations_at_candidate, std_iterations_at_candidate, all_cands

    candplot = plt.figure(1)
    candline = plt.plot(crs_iteration,  crs_count_at_iteration)
    # candline3 = plt.plot(rs_iteration, rs_count_at_iteration)
    plt.setp(candline, linewidth=1, color='red', marker='o')
    # plt.setp(candline3, linewidth=1, color='black', marker='.')
    for num in range(len(iters)):
        plt.plot(iters[num], cands[num],linewidth=1, color='green', marker='o', alpha=0.7)
    plt.xlabel("Iterations")
    plt.ylabel("Candidates Found")
    plt.title("{}: Candidates vs Iterations".format("Preliminary SigOpt Comparison with Chem Rules"))
    plt.show()

# SUBSPACE TECHNIQUE
def sigopt_one(fitness_evaluator=eval_fitness_complex, experiment_name = "Perovskite trial",
            token=sig, experiment_id=None, dimensions = [(0,51),(0,51),(0,6)], sleeptime = 45,
               n=None):

    conn = Connection(client_token=token)

    # VALUES FOR KEEPING STATS ON RUNS
    iscand = False

    # CREATE NEW EXPERIMENT OR RERUN
    if experiment_id == None:
        experiment = conn.experiments().create(
            name=experiment_name,
            parameters=[
                dict(name='A', type='int', bounds=dict(min=dimensions[0][0], max=dimensions[0][1])),
                dict(name='B', type='int', bounds=dict(min=dimensions[1][0], max=dimensions[1][1])),
                dict(name='anion', type='int', bounds=dict(min=dimensions[2][0], max=dimensions[2][1]))], )
    else:
        class ExperimentClass():
            id = experiment_id

        experiment = ExperimentClass()

    # CREATE SIGOPT SUGGESTION
    guess = conn.experiments(experiment.id).suggestions().create()
    tuple_guess = (guess.assignments['A'], guess.assignments['B'], guess.assignments['anion'])
    score = evaluate_model(guess.assignments, fitness_evaluator)
    time.sleep(sleeptime)

    mod_entry = mendeleev2atomic(tuple_guess)

    if score == 30:
        if mod_entry in GOOD_CANDS_LS:
            iscand = True

    observation = conn.experiments(experiment.id).observations().create(
        suggestion=guess.id,
        value=score)

    true_iteration = list_sigopt_observations(experiment.id, token=token)

    print "EXP:", experiment.id, "CAL:", true_iteration, "-", n, "SCORE:", score

    return iscand, mod_entry, experiment.id, true_iteration
def save_multiarm(exps, filename,results=None):
    with open(filename, 'a') as text_file:
        if results != None:
            day = str(datetime.date.today())
            thetime = datetime.datetime.now().time().isoformat()
            text_file.write("\n\nRUN COMPLETED: {} - {}".format(day, thetime))
            text_file.write("\nRESULTS IN ORDER:{}".format(results))
        for expid in exps.keys():
            text_file.write("\nSUBSPACE {} RESULTS:{}".format(expid, exps[expid]))()
def timeout_handler(signum, frame):
    print "RUNTIME ERROR"
    raise RuntimeError("timeout limit reached, closing suggestion and retrying")
def worker_multiarm():

    timeout = 5

    dims = []
    ranges = [(0, 12), (12, 25), (25, 38), (38, 51)]
    for k in ranges:
        for l in ranges:
            dims.append([k, l, (0, 6)])

    # SETUP
    exps = {}
    n = 1
    for i, dim in enumerate(dims):
        running = True
        expid = 0

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        while running:
            try:
                iscand, cand, expid, iter = sigopt_one(token=runtoken, dimensions=dim, sleeptime=runsleep,
                                                       experiment_name="Perovskite Trial Block {}".format(i),
                                                       n=n)
            except ConnectionException, E:
                print "HANDLING CONNECTION EXCEPTION"
                time.sleep(runsleep)
                continue

            except ApiException, A:
                print "HANDLING API EXCEPTION {}".format(A)
                break

            except RuntimeError, R:
                print "HANDLING OTHER ERROR {}".format(R)
                signal.alarm(timeout)
                continue

            running = False
            n += 1

        exps[expid] = {"dim": dim, "subspace_results": [], "candidates": [], "errors":[]}

    # RUN
    results = [0]
    save_interval = 160

    while n<1000:
        for expid in exps.keys():
            running = True
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            while running:
                try:
                    iscand, cand, expid, iter = sigopt_one(token=runtoken, experiment_id=expid,
                                                           sleeptime=runsleep,
                                                           n=n)
                except ConnectionException, E:
                    exps[expid]["errors"].append(str(E))
                    print "HANDLING CONNECTION EXCEPTION"
                    continue

                except ApiException, A:
                    exps[expid]["errors"].append(str(A))
                    print "HANDLING API EXCEPTION {}".format(A)
                    break

                except RuntimeError, R:
                    exps[expid]["errors"].append(str(R))
                    print "HANDLING OTHER ERROR {}".format(R)
                    try:
                        conn = Connection(client_token=runtoken)
                        x = conn.experiments(expid).suggestions().fetch(state="open")
                        for sug in x.iterate_pages():
                            conn.experiments(expid).suggestions(sug.id).update(state="closed")
                        signal.alarm(timeout)
                        break
                    except:
                        break

                n += 1
                running = False
                if iscand == True:
                    if cand not in exps[expid]["candidates"]:
                        results.append(n)
                        exps[expid]["subspace_results"].append(iter)
                        exps[expid]["candidates"].append(cand)

            if n % save_interval == 0:
                save_multiarm(exps, 'temp_results{}.txt'.format(n), results=results)

    save_multiarm(exps, 'results.txt', results=results)

class ChooseNextArm(object):
    def __init__(self, exps):
        self.exps = exps
        self.expids = self.exps.keys()

    def choose(self, arms):
    # arms should be in the form arms = {expid1:prob1, expid2:prob2, etc.}
        pass


    def epsilon_greedy(self):
    # Updates inidividual arm likelihoods based on solution frequency
        freq = [len(expid["subspace_results"]) for expid in self.expids]
        # arms = dict(zip(self.expids,freq))
        # next_choice = self.choose(arms)
        # return next_choice
        pass

    def random(self):
    # Arm likelihoods are equal
        pass


if __name__== "__main__":

    runtoken = dev
    runsleep = 30
    sigopt_trial(cands=3, experiment_name="Tracking Single O2F", trial_num="O2F", token=runtoken, tracking=False)













