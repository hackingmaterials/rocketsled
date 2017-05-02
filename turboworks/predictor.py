



from turboworks.optimize import OptTask, Dtypes
from itertools import product
import random
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

dtypes = Dtypes()

def calculate_discrete_space(dims):

    total_dimspace = []

    for dim in dims:
        if type(dim[0]) in dtypes.ints:
            # Then the dimension is of the form (lower, upper)
            lower = dim[0]
            upper = dim[1]
            dimspace = list(range(lower, upper + 1))
        elif type(dim[0]) in dtypes.floats:
            raise ValueError("The dimension is a float. The dimension space is infinite.")
        else:  # The dimension is a discrete finite string list
            dimspace = dim
        total_dimspace.append(dimspace)

    return [[x] for x in total_dimspace[0]] if len(dims) == 1 else list(product(*total_dimspace))


def gen_randoms(x_dim, n_guesses):

    randoms = []
    for i in range(n_guesses):
        randx = []
        for dim in x_dim:
            dim_type = type(dim[0])
            if dim_type in dtypes.discrete:
                if dim_type in dtypes.ints:
                    randx.append(random.randint(dim[0], dim[1]))
                elif dim_type in dtypes.others:
                    randx.append(random.choice(dim))
        randoms.append(randx)

    return randoms

def obj_function (X):

    for x in X:
        score_multiplier = 1
        if x[2] == 'red':
            score_multiplier = 2
        elif x[2] == 'blue':
            score_multiplier = 3
        elif x[2] == 'green':
            score_multiplier = 4

        return x[0]/x[1] * score_multiplier

def get_z (x):
    return [x[0]+x[1], x[1]/x[0]]


# x dims
dims = [(1,10),(12,15), ['red', 'green', 'blue', 'orange']]
space = calculate_discrete_space(dims)
X = gen_randoms(dims, 100)
Z = [get_z(x) for x in X]

X_tot = []
for i in range(len(X)):
    X_tot.append(X[i] + Z[i])


y = obj_function(X_tot)

# go from string to
le = LabelEncoder()
enc = OneHotEncoder()
le.fit_transform(dims)

untried = [datum for datum in space if datum not in X]

reg = linear_model.LinearRegression()
reg.fit(X,y)
reg.predict(untried)

