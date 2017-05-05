



from turboworks.optimize import OptTask, Dtypes
from itertools import product
import random
from sklearn import linear_model, svm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from skopt.space import Space

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

def obj_function (X):

    y = []
    for x in X:
        score_multiplier = 1
        if x[2] == 'red':
            score_multiplier = 2
        elif x[2] == 'blue':
            score_multiplier = 3
        elif x[2] == 'green':
            score_multiplier = 4

        y.append(x[0] * x[1] * score_multiplier)
    return y

def get_z (x):
    return [x[0]+x[1], x[1]/x[0]]


# x dims
# dims = [(1,10),(12,15), ['red', 'green', 'blue', 'orange']]
dims = [(1,10), (12,15)]
space = calculate_discrete_space(dims)
X = random.sample(space, 15)
Z = [get_z(x) for x in X]
X_tot = [list(X[i]) + Z[i] for i in range(len(X))]

X_untried = [list(x) for x in space if x not in X]
Z_untried = [get_z(x) for x in X_untried]
X_tot_untried = [X_untried[i] + Z_untried[i] for i in range(len(X_untried))]

y = obj_function(X_tot)


# todo: go from string to sklearnable dimension with LabelEncoder and OneHotEncoder
def preprocess(X, dims):
    """
    Transforms categorical dimensions into integer dimensions for prediction via predictor. 
    """
    pass


def predictor(X, y, space, model, n_points = 10000, minimize=True):
    """
    Scikit-learn compatible model for stepwise optimization. It uses a regressive predictor evaluated on all possible 
    remaining points in a discrete space. OptTask Z and X are abstracted.
    
    Args:
        X ([list]): List of vectors containing input training data.
        y (list): List of scalars containing output training data.
        space ([list]): List of vectors containing all possible inputs. Should be preprocessed before being passed to
            predictor function.
        model (sklearn model): The regressor used for predicting the next best guess.
        n_points (int): The number of points in space to predict over.
        minimize (bool): Makes predictor return the guess which maximizes the predicted objective function output.
            Else maximizes the predicted objective function output.  
            
    Returns:
        (list) A vector which is predicted to minimize (or maximize) the objective function. This vector contains 
            extra 'z' features which will need to be discarded in postprocessing. However, 'x' and 'z' information
            is guaranteed to match. 
        
    """

    #todo: currently only working with integer/float dimensions

    n_points = len(space) if n_points > len(space) else n_points
    X_predict = random.sample(space, n_points)
    model.fit(X, y)
    values = model.predict(X_predict).tolist()
    evaluator = min if minimize else max
    i = values.index(evaluator(values))
    return X_predict[i]


print predictor(X_tot, y, X_tot_untried, svm.SVR())