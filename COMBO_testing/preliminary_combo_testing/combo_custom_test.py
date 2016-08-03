
"""
Testing the ability of combo on a random BB function
"""
import numpy as np
import combo
from turboworks.discrete_spacify import calculate_discrete_space




# INTEGER

dimensions = [(1,10),(1,10)]
X = calculate_discrete_space(dimensions)

class BBclass:
    def __init__(self):
        self.X = X

    def BBfun(self,x):
        return x[0]*x[1]/3.292884

    def __call__(self, action):
        return self.BBfun(self.X[action])

policy = combo.search.discrete.policy(test_X=np.asarray(X))
policy.set_seed(0)
res = policy.random_search(max_num_probes=5, simulator=BBclass())

res = policy.bayes_search(max_num_probes=20, simulator=BBclass(), score='TS',
                          interval=20, num_rand_basis=5000)

print "best_fx"
print res.export_sequence_best_fx()[0]

print "best_actions"
print res.export_sequence_best_fx()[1]



#   CATEGORICAL --- doesnt work

# dim_cat = [["red", "green", "blue"], ["dog","cat","mouse"]]
# X_cat = calculate_discrete_space(dim_cat)
#
# class BB_cat:
#     def __init__(self):
#         self.X = X_cat
#
#     def BBfun_cat(self, x):
#         score = 0
#         if x[0] == "red":
#             score+=5
#         elif x[0] == "green":
#             score+=2.5
#         elif x[0] == "blue":
#             score -=2.5
#         if x[1] == "dog":
#             score+=10
#         return score
#
#     def __call__(self, action):
#         return self.BBfun_cat(self.X[action])
#
# policy_cat = combo.search.discrete.policy(test_X=np.asarray(X_cat))
# policy_cat.set_seed(0)
# res2 = policy_cat.random_search(max_num_probes=5, simulator=BB_cat())
#
# res3 = policy_cat.bayes_search(max_num_probes=3, simulator=BB_cat(), score='TS',
#                           interval=20, num_rand_basis=5000)
#
# s = BB_cat()