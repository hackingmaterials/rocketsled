import multiprocessing
from multiprocessing import Manager


def multiprocessing_mendeleev_comparisons(iter_num=5, run_num=5, initial_guessing="random"):

    skopt_cands = Manager().list()
    skopt_iters = Manager().list()
    skopt_times = Manager().list()
    combo_cands = Manager().list()
    combo_iters = Manager().list()
    combo_times = Manager().list()
    iterations = Manager().list()

    # A single run to be optimized (either skopt of combo)
    def job(type):
        if type == "skopt":
            fun = mendeleev_integer_optimization_line_and_timing
        elif type=="combo":
            fun = mendeleev_integer_optimization_combo_line_and_timing

        if initial_guessing == "random":
            initial_guess = dummy_minimize([name_index, name_index, anion_names])
        else:
            initial_guess = initial_guessing

        cand_iter, cand_count_at_iter, iterations_single, times = fun(guess=initial_guess, iterations=iter_num)

        if type == "skopt":
            skopt_cands.append(cand_count_at_iter)
            skopt_iters.append(cand_iter)
            skopt_times.append(times)
        elif type == "combo":
            combo_cands.append(cand_count_at_iter)
            combo_iters.append(cand_iter)
            combo_times.append(times)
        iterations.append(iterations_single)

    jobs = []

    #TODO: use Pool class instead of Process (pool reallocates processes dynamically as they are finished)

    for i in range(run_num):
        # p_combo = multiprocessing.Process(target=job, args=("combo",))
        p_skopt = multiprocessing.Process(target=job, args=("skopt",))
        # jobs.append(p_combo)
        jobs.append(p_skopt)
        # p_combo.start()
        p_skopt.start()

    for proc in jobs:
        proc.join()

    print "done optimizing the jobs in parallel"

    skopt_cands = list(skopt_cands)
    skopt_iters = list(skopt_iters)
    skopt_times = list(skopt_times)
    combo_cands = list(combo_cands)
    combo_iters = list(combo_iters)
    combo_times = list(combo_times)

    save_and_show(skopt_iters, skopt_cands, skopt_times, combo_iters, combo_cands, combo_times, iterations)