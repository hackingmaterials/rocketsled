

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from turboworks.optimize import Dtypes

dims = [(0, 10), (11, 20), ["red", "blue", "green", "orange"], (1,10), ["cat", "dog", "rat"]]

X = [[1, 18, "red", 9, "dog"], [2, 19, "blue", 8, "cat"], [1, 18, "orange", 7, "rat"]]

dtypes = Dtypes()

n_cats = 0

bin_info = []

for i, dim in enumerate(dims):
    if type(dim[0]) in dtypes.others:

        cats = [0]*len(X)
        for j, x in enumerate(X):
            cats[j] = x[i - n_cats]

        forward_map = {k: v for v, k in enumerate(dim)}

        try:
            # Python 2.x
            inverse_map = {v: k for k, v in forward_map.iteritems()}

        except:
            # Python 3.x
            inverse_map = {v: k for k, v in forward_map.items()}


        lb = LabelBinarizer()
        binary = lb.fit_transform([forward_map[v] for v in dim])

        dim_info = {'index':i, 'onehot_length':binary[0], 'lb':lb, 'inv_map': inverse_map}
        bin_info.append(dim_info)


        for j, x in enumerate(X):
            x.remove(x[i - n_cats])
            x += list(binary[j])

        n_cats += 1


print X

# new_x = [1, 18, 9, 1, 0, 0, 1, 0, 0]
# exported_new_x = []
# for i, xi in enumerate(new_x):
#     if i in index_map:
#         for lb
#         xi = inverse_map
#     else:
#         exported_new_x.append(xi)
