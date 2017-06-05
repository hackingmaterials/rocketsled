import random
import numpy as np

# Simulates a random "take one out" guessing performance algorithm for perovskite searching

n = 18928
X = list(range(n))
solutions = random.sample(X, 20)

I_tot = []
n_runs = 50

for run in range(n_runs):
    print "run", run
    I = []
    for i in range(n):
        guess = random.choice(X)
        X.remove(guess)
        if guess in solutions:
            I.append(i)
    X = list(range(n))
    I_tot.append(I)

I_mean = np.mean(I_tot, axis=0).tolist()

print I_mean

diffs = [I_mean[i+1] - I_mean[i] for i in range(len(I_mean) - 1)]
avg = np.mean(diffs)
print avg

results = [763.46, 1682.28, 2685.18, 3536.56, 4604.58, 5920.56, 6555.42, 7667.04, 8412.74, 9228.02, 10016.54, 10796.92,
           11655.7, 12701.22, 13784.92, 14490.68, 15392.34, 16295.12, 17094.34, 18045.58]
results_avg = 909.585263158



