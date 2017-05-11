

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from turboworks.optimize import Dtypes

dims = [(0, 10), (11, 20), ["red", "blue", "green", "orange"], (1,10), ["cat", "dog", "rat"]]

X = [[1, 18, "red", 9, "dog"], [2, 19, "blue", 8, "cat"], [1, 18, "orange", 7, "dog"]]

dtypes = Dtypes()

n_cats = 0

index_map = {}
inverse_maps = []
lbs = []

for i, dim in enumerate(dims):
    if type(dim[0]) in dtypes.others:
        cats = [0]*len(X)
        for j, x in enumerate(X):
            cats[j] = x[i - n_cats]

        map = {k: v for v, k in enumerate(cats)}

        try:
            inverse_map = {v: k for k, v in map.iteritems()}

        except:
            inverse_map = {v: k for k, v in map.items()}

        inverse_maps.append(inverse_map)

        lb = LabelBinarizer()
        binary = lb.fit_transform([map[v] for v in cats])

        for j, x in enumerate(X):
            x.remove(x[i - n_cats])
            x += list(binary[j])

        index_map[i] = len(binary[0])
        n_cats += 1

# new_x = [1, 18, 9, 1, 0, 0, 1, 0, 0]
# exported_new_x = []
# for i, xi in enumerate(new_x):
#     if i in index_map:
#         for lb
#         xi = inverse_map
#     else:
#         exported_new_x.append(xi)
