

from sklearn.preprocessing import LabelBinarizer
from turboworks.optimize import Dtypes
import numpy as np

# dims = [(0, 10), (11, 20), ["red", "blue", "green", "orange"], (1, 10), ["cat", "dog", "rat"], (1, 15)]
# X = [[1, 18, "red", 9, "dog", 8], [2, 19, "blue", 8, "cat", 6], [1, 18, "orange", 7, "rat", 2]]

# dims = [["red", "blue", "green", "orange"], ["cat", "dog", "rat"]]
# X = [["red", "rat"], ["green", "cat"]]

dims = [(1,34), (100,200), (1.24353, 932423.2)]
XZ = [[22, 124, 15.332, "red"], [12, 101, 16.2929, "green"], [13, 199, 100.001, "blue"]]

dtypes = Dtypes()


n_cats = 0
bin_info = []


for i, dim in enumerate(dims):
    if type(dim[0]) in dtypes.others:

        cats = [0]*len(XZ)
        for j, xz in enumerate(XZ):
            cats[j] = xz[i - n_cats]

        print "cats",  cats

        forward_map = {k: v for v, k in enumerate(dim)}

        print "forward dims", forward_map

        inverse_map = {v: k for k, v in forward_map.items()}

        lb = LabelBinarizer()

        lb.fit([forward_map[v] for v in dim])
        binary = lb.transform([forward_map[v] for v in cats])

        print "binary for", cats, " ", binary

        # print "forward transform input", [forward_map[v] for v in dim]
        # print "forward transform output", binary
        #
        # print "inverse transformed", lb.inverse_transform(np.asarray([binary[0]]))

        for j, x in enumerate(XZ):
            del(x[i - n_cats])
            x += list(binary[j])

        dim_info = {'lb': lb, 'inv_map': inverse_map, 'binary_len': len(binary[0])}
        print dim_info

        bin_info.append(dim_info)

        n_cats += 1
    print XZ

# new_x = [2, 19, 8, 6, 0, 1, 0, 0, 1, 0, 0]
# new_x = [1, 0, 0, 0, 0, 0, 1]
new_x = [33, 104, 18.9293912]

cat_index = 0
tot_bin_len = 0

original_len = len(dims)
static_len = original_len - n_cats

exported_new_x = []

for i, dim in enumerate(dims):
    if type(dim[0]) in dtypes.others:
        dim_info = bin_info[cat_index]

        binary_len = dim_info['binary_len']
        lb = dim_info['lb']
        inverse_map = dim_info['inv_map']

        start = static_len + tot_bin_len
        end = start + binary_len
        binary = new_x[start:end]

        int_value = lb.inverse_transform(np.asarray([binary]))[0]
        cat_value = inverse_map[int_value]
        exported_new_x.append(cat_value)

        cat_index += 1
        tot_bin_len += binary_len


    else:
        exported_new_x.append(new_x[i - cat_index])

print exported_new_x