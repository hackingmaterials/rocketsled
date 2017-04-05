import itertools
from turboworks.references import dtypes
import time
from matplotlib import pyplot as plot
from pymongo import MongoClient
from gridfs import GridFS
from sys import getsizeof

def calculate_discrete_space(dims):
    """
    Calculates all entries in a discrete space.

    Example:

        >>> dims = [(1,2), ["red","blue"]]
        >>> space = _calculate_discrete_space(dims)
        >>> space
        [(1, 'red'), (1, 'blue'), (2, 'red'), (2, 'blue')]

    Args:
        dims ([tuple]): dimensions of the search space.

    Returns:
        ([list]) all possible combinations inside the discrete space
    """

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

    return [[x] for x in total_dimspace[0]] if len(dims ) ==1 else list(itertools.product(*total_dimspace))

if __name__ == "__main__":

    times = []
    byte_sizes = []
    size = [2,4,6,8,10,20,30,50,70,100,110,120,130,140,150,160,170,180,190,200,210,230,245,260]
    size_real = [z**3 for z in size]

    q = 10


    for i in size[0:q]:
        print "running ", i
        dims = [(1,i), (1,i), (1,i)]

        start = time.time()
        dimspace = calculate_discrete_space(dims)
        end = time.time()

        byte_sizes.append(getsizeof(dimspace))

        times.append(end-start)

    print byte_sizes


    dimspace1 = dimspace[0]
    conn = MongoClient('localhost', 27017)
    db = conn.tester
    db.tw_example.insert_many([{'z':dimspace1[i]} for i in dimspace1])
    db.tw_example.insert_one({'available_z':dimspace1})
    # fs = GridFS(db, collection="tw_example")
    # with fs.new_file() as fp:
    #     fp.write(dimspace1)
    #
    # with fs.get(fp._id) as fp_read:
    #     dimspace1_fromdb = fp_read.read()
    #
    # print type(dimspace1_fromdb)
    # print dimspace1


