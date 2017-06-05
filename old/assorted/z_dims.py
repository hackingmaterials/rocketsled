from turboworks.optimize import Dtypes

dtypes = Dtypes()

Z = [["red"], ["green"], ["orange"]]

dims = [(z, z) for z in Z[0]]

for i, dim in enumerate(dims):
    cat_values = []
    for z in Z:
        if type(z[i]) in dtypes.others:
            # the dimension is categorical
            if z[i] not in cat_values:
                cat_values.append(z[i])
                dims[i] = cat_values

print dims