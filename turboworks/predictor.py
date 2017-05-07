import random

# todo: go from string to sklearnable dimension with LabelEncoder and OneHotEncoder
def preprocess(X, dims):
    """
    Transforms categorical dimensions into integer dimensions for prediction via predictor. 
    """
    pass


def sk_predictor(X, y, space, model, n_points = 10000, minimize=True):
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