from perovskite_task import *
from sigopt import Connection
from sigopt.exception import ConnectionException, ApiException
import time
import signal
import csv


dev = "BRSOTAQXVWFBVRLFXLEJYIFXOPKXYXQAUUAXFECVHKJPDPRN"
sig = "VEXZNUGYFSANVIWWDECDVMUELNFIAVWNJXQDSXALTMBOKAJQ"

def timeout_handler(signum, frame):
    print "RUNTIME ERROR"
    raise RuntimeError("timeout limit reached, closing suggestion and retrying")

def num_observations(experiment_id, token=sig):
    #returns number of observations for given experiment
    conn = Connection(client_token=token)
    observations_list = conn.experiments(experiment_id).fetch().progress.observation_count
    return observations_list

class trial(object):
    '''The next version of this file will have all classes derived from this trial'''
    def __init__(self):
        pass

class sigopt_responsive(object):
    def __init__(self, cand_limit=1, fitness_evaluator=eval_fitness_complex, experiment_name = "PEROVSKITES",
                 token=sig, exhaustive=False, cascading=True, radius=1, multiplier = 1.5, dimensions = None,
                 trial_num="None", tracking=False, sleeptime = 30):

        # Argument variables
        self.cand_limit = cand_limit
        self.fitness_evaluator = fitness_evaluator
        self.experiment_name = experiment_name
        self.token = token
        self.exhaustive = exhaustive
        self.radius = radius
        self.cascading = cascading
        self.trial_num = trial_num
        self.tracking = tracking
        self.multiplier = multiplier
        self.sleeptime = sleeptime
        self.dimensions = dimensions
        self.experiment_id=0

        if dimensions == None:
            self.dimensions = [(0,51),(0,51),(0,6)]

        # Meta variables
        self.conn = Connection(client_token=self.token)
        self.timeout = 1200
        self.save_interval = 500
        self.iter_limit = 4000
        self.hitscore = 30
        self.scale = 1.0

        # Trackers and counters
        self.candidates = []
        self.explored = []
        self.calculations = []
        self.errors = []
        self.simulated_iteration = 0
        self.sigopt_iteration = 0
        self.blacklist = []


    def start(self):
        '''Begins experiment'''
        name = self.experiment_name
        experiment = self.conn.experiments().create(
            name=name,
            parameters=[
                dict(name='A', type='int', bounds=dict(min=self.dimensions[0][0], max=self.dimensions[0][1])),
                dict(name='B', type='int', bounds=dict(min=self.dimensions[1][0], max=self.dimensions[1][1])),
                dict(name='X', type='int', bounds=dict(min=self.dimensions[2][0], max=self.dimensions[2][1]))], )
        print "experiment", experiment.id, "created"
        return experiment

    def set(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)

    def evaluate(self, tuple_guess):
        '''Scores a tuple of form (A,B,X)'''
        q = mendeleev_rank_to_data(tuple_guess)[0]
        fit_score = self.fitness_evaluator(q['gap_dir'], q['gap_ind'], q['heat_of_formation'],
                                      q['vb_dir'], q['cb_dir'], q['vb_ind'], q['cb_ind'])
        return fit_score*self.scale

    def make_tracker(self):
        '''For use with tracking sigopt trials'''
        filename = '{}_{}_mendeleev_tracker.csv'.format(self.experiment_name, self.trial_num)
        with open(filename, 'w') as csvfile:
            fieldnames = ['A', 'B', 'X']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            assignments = {}
            assignments['A'] = 'A'
            assignments['B'] = 'B'
            assignments['X'] = 'X'
            writer.writerow(assignments)

    def use_tracker(self, what):
        '''For use with tracking sigopt trials'''
        filename = '{}_{}_mendeleev_tracker.csv'.format(self.experiment_name, self.trial_num)
        with open(filename, 'a') as csvfile:
            fieldnames = ['A', 'B', 'X']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            assignments = {}
            assignments['A'] = what[0]
            assignments['B'] = what[1]
            assignments['X'] = what[2]
            writer.writerow(assignments)

    def check(self, tuple_guess):
        '''Check to see if tuple_guess (in mendeleev rank form) is a good candidate'''
        is_new_candidate = False
        mod_entry = mendeleev2atomic(tuple_guess)
        if mod_entry in GOOD_CANDS_LS_WRONG and mod_entry not in self.candidates:
            self.candidates.append(mod_entry)
            self.calculations.append(self.simulated_iteration)
            is_new_candidate = True
        return is_new_candidate

    def sphere(self, tuple_guess):
        '''Create 3D sphere of points around point tuple_guess'''
        A_ion = tuple_guess[0]
        B_ion = tuple_guess[1]
        X_ion = tuple_guess[2]
        r = self.radius

        if A_ion - r < 0:
            A_ion_lower = 0
        else:
            A_ion_lower = A_ion - r
        if A_ion + r > 52:
            A_ion_higher = 52
        else:
            A_ion_higher = A_ion + r

        if B_ion - r < 0:
            B_ion_lower = 0
        else:
            B_ion_lower = B_ion - r
        if B_ion + r > 52:
            B_ion_higher = 52
        else:
            B_ion_higher = B_ion + r

        if X_ion - r < 0:
            X_ion_lower = 0
        else:
            X_ion_lower = X_ion - r
        if X_ion + r > 6:
            X_ion_higher = 6
        else:
            X_ion_higher = X_ion + r

        sphere_dim = [(A_ion_lower, A_ion_higher), (B_ion_lower, B_ion_higher), (X_ion_lower, X_ion_higher)]

        sphere = calculate_discrete_space(sphere_dim)
        print "CREATING SPHERE"

        return sphere

    def blacklist_points(self, points):
        '''Uniquely add points to blacklist'''
        print "BLACKLISTING SPHERE"
        for point in points:
            if point not in self.blacklist:
                self.blacklist.append(point)

    def record(self, guess=None, tuple_guess = None):
        '''Record point either as a guess from prediction or as other observation'''

        were_points_blacklisted = False
        if guess!=None and tuple_guess is None:
            tuple_guess = (guess.assignments['A'], guess.assignments['B'], guess.assignments['X'])

        self.simulated_iteration += 1
        score = self.evaluate(tuple_guess)
        print self.simulated_iteration, "RECORDED", tuple_guess, "SCORE", score, "SCALE", self.scale
        self.sigopt_iteration = num_observations((self.experiment_id), token=self.token)
        self.explored.append(tuple_guess)
        if self.tracking:
            self.use_tracker(tuple_guess)

        if score == self.hitscore:
            if self.check(tuple_guess):
                atomic_guess = mendeleev2atomic(tuple_guess)
                print "CANDIDATE FOUND: MENDELEEV", tuple_guess, "ATOMIC", atomic_guess, "\n"

                if self.exhaustive:
                    self.exhaustive_search(tuple_guess)
                    were_points_blacklisted = True

        if guess is None:
            assignments = {'A': tuple_guess[0], 'B': tuple_guess[1], 'X': tuple_guess[2]}
            observation = self.conn.experiments(self.experiment_id).observations().create(
                assignments=assignments, value=score)
        else:
            observation = self.conn.experiments(self.experiment_id).observations().create(
                suggestion=guess.id, value=score)

        return were_points_blacklisted

    def exhaustive_search(self, tuple_guess):
        '''Blacklist points in the sphere, then record all points in the sphere'''
        space = self.sphere(tuple_guess)
        self.blacklist_points(space)

        unexplored = [x for x in space if x not in self.explored]
        print "EXHAUSTIVELY SEARCHING", len(unexplored), "POINTS"
        # Record points not already explored
        for i, point in enumerate(unexplored):
            self.record(tuple_guess=point)
            print "SEARCHED POINT ", i+1, "OF", len(unexplored), "POINTS"

    def upgrade_whitelist(self):
        for observation in self.conn.experiments(self.experiment_id).observations().fetch().iterate_pages():
            a = observation.assignments
            tuple_guess = (a['A'], a['B'], a['X'])
            if tuple_guess not in self.blacklist:
                score = self.evaluate(tuple_guess)
                self.conn.experiments(self.experiment_id).observations(observation.id).update(value = score)

    def text_save(self, filename):
        with open(filename, 'a') as text_file:
            day = str(datetime.date.today())
            time = datetime.datetime.now().time().isoformat()
            text_file.write("\n\nRUN COMPLETED: {} - {}".format(day, time))
            text_file.write("\nEXPERIMENT NAME: {}".format(self.experiment_name))
            text_file.write("\nEXPERIMENT ID: {}".format(self.experiment_id))
            text_file.write("\nCANDIDATES: {}".format(self.candidates))
            text_file.write("\nITERATIONS: {}".format(self.calculations))
            text_file.write("\nSIMULATED ITERATIONS: {}".format(self.simulated_iteration))
            text_file.write("\nSIGOPT ITERATIONS: {}".format(self.sigopt_iteration))
            text_file.write("\nERRORS: {}".format(self.errors))


    def run(self):

        if self.experiment_id==0:
            self.experiment_id = self.start().id
        if self.tracking:
            self.make_tracker()

        while len(self.candidates) != self.cand_limit and self.sigopt_iteration < self.iter_limit:
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.timeout)
                guess = self.conn.experiments(self.experiment_id).suggestions().create()
                time.sleep(self.sleeptime)
                blacklist_was_updated =  self.record(guess = guess)
                if blacklist_was_updated:
                    self.scale *= self.multiplier
                    self.hitscore *= self.multiplier
                    self.upgrade_whitelist()


            except ConnectionException, E:
                print "HANDLING CONNECTION EXCEPTION"
                time.sleep(self.sleeptime)
                self.simulated_iteration -= 1
                self.errors.append("Calculation {}, Error {} \n".format(self.simulated_iteration, E))
                continue

            except ApiException, A:
                print "HANDLING API EXCEPTION {}".format(A)
                self.errors.append("Calculation {}, Error {} \n".format(self.simulated_iteration, A))
                break

            except RuntimeError, R:
                print "HANDLING OTHER ERROR {}".format(R)
                signal.alarm(self.timeout)
                self.errors.append("Calculation {}, Error {} \n".format(self.simulated_iteration, R))
                continue

            if self.sigopt_iteration!=0:
                if self.sigopt_iteration % self.save_interval == 0:
                    prev = self.sigopt_iteration
                    self.text_save(filename="{}_{}_temp_trial.txt".format(self.experiment_name,
                                                                         self.trial_num, self.sigopt_iteration))

        self.text_save(filename="{}_{}_final.txt".format(self.experiment_name, self.trial_num, self.sigopt_iteration))
        print "FINISHED. FOUND {} CANDIDATES \n {}".format(len(self.candidates), self.candidates)


if __name__ == "__main__":
    trial = sigopt_responsive(cand_limit=20, experiment_name = "responsive-m3.0r2",
                 token=sig, exhaustive=True, cascading=True, radius=2, multiplier = 3.0, dimensions = None,
                 trial_num="5", tracking=True, sleeptime = 30)

    trial.run()

