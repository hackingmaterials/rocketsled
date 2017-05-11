

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from turboworks.optimize import Dtypes

dims = [(0, 10), (11, 20), ["red", "blue", "green", "orange"], (1,10), ["cat", "dog", "rat"]]

X = [[1, 18, "red", 9, "dog"], [2, 19, "blue", 8, "cat"], [1, 18, "orange", 7, "dog"]]

dtypes = Dtypes()

for i, dim in enumerate(dims):
    if type(dim[0]) in dtypes.others:
        col = [0]*len(X)
        for j, x in enumerate(X):
            col[j] = x[i]

        le = LabelEncoder()
        le.fit(dim)
        l_encoded = le.transform(col)

        ohe = OneHotEncoder()
        oh_encoded = ohe.fit_transform(l_encoded)


        print col
        print l_encoded
        print oh_encoded



