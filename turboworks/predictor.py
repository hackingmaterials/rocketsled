



from turboworks.optimize import OptTask, Dtypes
from itertools import product
import random
from sklearn import linear_model
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from skopt.space import Space

dtypes = Dtypes()

# todo: go from string to sklearnable dimension with LabelEncoder and OneHotEncoder
def preprocess(X, dims):
    """
    Transforms categorical dimensions into integer dimensions for prediction via predictor. 
    """
    pass


def define_space(dims, get_z, n_points=10000):
    """
    Determine all possible predictable points within the space. Must generate matching X and Z guesses. 
    """
    pass

def predictor(X, y, space, model, minimize=True):
    """
    Scikit-learn compatible model for stepwise optimization. It uses a regressive predictor evaluated on all possible 
    remaining points in a discrete space. OptTask Z and X are abstracted.
    
    Args:
        X ([list]): List of vectors containing input training data.
        y (list): List of scalars containing output training data.
        space ([list]): List of vectors containing all possible inputs. Should be preprocessed before being passed to
            predictor function.
        model (sklearn model): The regressor used for predicting the next best guess. 
        minimize (bool): Makes predictor return the guess which maximizes the predicted objective function output.
            Else maximizes the predicted objective function output.  
            
    Returns:
        (list) A vector which is predicted to minimize (or maximize) the objective function. This vector contains 
            extra 'z' features which will need to be discarded in postprocessing. However, 'x' and 'z' information
            is guaranteed to match. 
        
    """

    X_untried = [point for point in space if point not in X]
    model.fit(X, y)
    values = model.predict(X_untried)
    evaluator = min if minimize else max
    i = values.index(evaluator(values))
    return X_untried[i]