"""
Utility functions for managing the turboworks databases.

    __init__         : allows custom MongoDB params (ie, store the DB somewhere besides default)
    _store           : put custom, ready to go JSON data directly into the turboworks database
    process          : processes a dictionary and list of relevant keys into a turboworks-friendly JSON format
    store            : processes a custom dictionary and stores its optimization-ready counterpart in the tw db
    clean            : deletes the entire collection
    count            : counts how many documents are in the collection
    query            : queries the DB based on typical pymongo syntax
    avg              : finds average
    min              : finds min
    max              : finds max
    back_up          : stores the entire collection in a backup collection
"""

import warnings
from pymongo import MongoClient
from numpy import ndarray
from numpy import mean as npmean
from references import dtypes

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


class OptDB(object):

    """
    Manages the turboworks databases.

    """

    def __init__(self, hostname='localhost', portnum=27017, dbname='turboworks',
                 collection='turboworks', opt_label="Unnamed"):
        """
        Allows configuration of the TurboWorks DB

        :param hostname: (str) host name of the MongoDB
        :param portnum: (str) port number of the DB
        :param dbname: (str) database name (creates it if nonexistent)
        :param collection: (str) _collection name (creates it if nonexistent)
        :param opt_label: (String) Used for keeping track of optimization labels. If this does not match the opt_label
        that OptTask is using, this data will not be used!
        """
        self.mongo = MongoClient(hostname, portnum)
        self.db = getattr(self.mongo,dbname)
        self.collection = getattr(self.db,collection)
        self.collection_string = collection

        self.opt_label = opt_label



    def _store(self, obj, opt_label=None):
        """
        Internal utility for storing turboworks-ready documents in the database.

        :param obj: A dict or list of dicts which can be used with turboworks db.
                for example: obj = {'z':[1,2,3], 'y':5}

        :param opt_label: (String) Used for keeping track of optimization labels. If this does not match the opt_label
        that OptTask is using, this data will not be used!

        :return: None
        """

        warning_list = "key {} exists in dictionary {}, but is not a list or array. \n" \
                       "this may cause errors internally in turboworks"
        warning_value = "{} must be a {}. Entry skipped. \n" \
                        "Consult turboworks.references.dtypes for a full list of supported datatypes."
        warning_key = "required key {} not in {}. Entry skipped. \n" \
                      " Stored dictionaries must have 'z' and 'y' to be stored in the turboworks db."
        warning_dict = "Object {} is {}, not dict. Entry skipped. \n " \
                       "Turboworks can only _store dictionaries in the database."
        error_list = "Object {} is {}, not list. Turboworks can only process a list of dictionaries."

        if isinstance(obj, dict):
            obj = [obj]

        if isinstance(obj, list):
            for d in obj:
                if isinstance(d, dict):
                    if 'z' in d.keys():
                        if 'y' in d.keys():
                            if isinstance(d['z'], list) or isinstance(d['z'], ndarray):
                                if type(d['y']) in dtypes.numbers:
                                    store_dict = {'z': d['z'], 'y': d['y']}
                                    if 'x' in d.keys():
                                        if isinstance(d['x'], list) or isinstance(d['x'], ndarray):
                                            store_dict['x'] = d['x']
                                        else:
                                            warnings.warn(warning_list.format('x', d))

                                    if opt_label is not None:
                                        store_dict['opt_label'] = opt_label

                                    self.collection.insert_one(store_dict)
                                else:
                                    warnings.warn(warning_value.format('y', 'number'))
                            else:
                                warnings.warn(warning_list.format('z', d))
                        else:
                            warnings.warn(warning_key.format('y', d))
                    else:
                        warnings.warn(warning_key.format('z', d))
                else:
                    warnings.warn(warning_dict.format(d, type(d)))
        else:
            raise TypeError(error_list.format(obj, type(obj)))

    def process(self, obj, z_keys, y_key, x_keys = None):
        """
        Construct a turboworks-ready list of documents from a single (or list of) dictionary-style objects.

        :param obj: (list of dicts or dict) the original documents containing optimization info to be stored
        :param z_keys: (list of strings) keys from which to construct z
        :param x_keys: (list of strings) keys from which to construct x
        :param y_key: (string) key to construct y
        :return: the processed document in turboworks-ready format
        """
        processed = []

        if isinstance(obj, dict):
            obj = [obj]

        if x_keys is None:
            x_keys = []

        for d in obj:
            doc = {}
            doc['z'] = [d[zk] for zk in z_keys]
            doc['x'] = [d[xk] for xk in x_keys]
            doc['y'] = d[y_key]
            processed.append(doc)
        return processed

    def store (self, obj, z_keys, y_key, x_keys = None, opt_label=None):
        """
        User function for putting an iterable of dict-style objects into the optimization db directly.

        :param obj: (iterable of dicts) Iterable structure containing data for optimization.
        :param z_keys: (list of Strings) Key values of the unique features (z) vector
        :param y_key: (String) Key value of the y scalar
        :param x_keys: (list of Strings) Key values of the extra features (x) vector.
        :param opt_label: (String) optimization label. Use if you are storing more than one set of optimization data
        in the db.
        :return: None
        """

        if opt_label == None:
            opt_label = self.opt_label

        processed = self.process(obj, z_keys, y_key, x_keys=x_keys)
        self._store(processed, opt_label)


    @property
    def subcollection(self):
        return self.collection.find({'opt_label': self.opt_label})

    @property
    def count(self):
        """
        Counts documents in the TurboWorks DB _collection (TurboWorks_collection)

        :return: (int) the total number of documents in the collection
        """

        return self.subcollection.count()

    @property
    def min(self):
        """
        Finds the minimum value of 'y' in the database.

        :return: (Result object) holds all information pertinent to the minimum 'y' value in the turboworks db.
        """

        Y = [y['y'] for y in self.subcollection]
        return Result(min(Y), self.subcollection)

    @property
    def max(self):
        """
        Finds the maximum value of 'y' in the database.

        :return: A Result object holding all information pertinent to the maximum 'y' value in the turboworks db.
        """

        Y = [y['y'] for y in self.subcollection]
        return Result(max(Y), self.subcollection)

    def query(self, querydict=None, print_to_console = False):
        """
        Queries documents via PyMongo's find()

        :param print_to_console: (boolean) determine whether all found matching docs should be
        printed to the console

        :param querydict: (dictionary) a dictionary query entry compatible with pymongo syntax

        :return docs: (list) a list of all documents matching the query
        """
        if querydict is None:
            querydict = {}

        querydict.update({'opt_label':self.opt_label})

        cursor = self.collection.find(querydict)
        print('Documents matching:        ', cursor.count())
        print('Documents:')

        if print_to_console:
            for document in cursor:
                    print(document)

        docs = [document for document in cursor]
        return docs

    def clean(self, clean_all=False, query=None):
        """
        Deletes all data in the TurboWorks DB collection

        :param clean_all: (Boolean) specify whether to clean all optimization data or just for the current opt_label
        :param query: (mongodb compatible dict) a filter which selects the documents to delete
        :return num_deleted: (int) number of documents deleted from the database
        """
        if query is None:
            query = {}

        if clean_all:
            pass
        else:
            if 'opt_label' not in query.keys():       # user did not specify an opt_label for cleaning
                query['opt_label'] = self.opt_label   # use the current one for the instance

        docs_removed = self.collection.delete_many(query)
        num_deleted = docs_removed.deleted_count

        try:
            print('Documents removed from {db}: {n}'.format(db=self.collection_string, n=num_deleted))
        except:
            print('Documents removed from {db}: {n}'.format(db=self.collection_string, n=num_deleted))
        print('Documents remaining: {n}'.format(n=self.collection.count()))
        return num_deleted

    def avg(self, var):
        """
        Finds the average of a vector or scalar across the database.

        :param var: (string) the variable to be averaged. Should be 'Z', or 'X' or 'Y'
            example: get_avg("Z")
        :return mean: (list) of average values across the db for each of the features in var
            example: db has two docs, {'X':[1,2,3]} and {'X':[2,4,6]}. Returns [1.5, 3, 4.5].
        """

        if var in ['X', 'Y', 'Z']:
            var = var.lower()

        X = [x[var] for x in self.subcollection]

        mean = npmean(X, 0).tolist()
        return mean

    def back_up(self, hostname='localhost', portnum=27017, dbname='turboworks',
                collection='backup'):
        """
        Transfers all data from the TurboWorks DB to another DB

        :param hostname: (str) host name of the MongoDB
        :param portnum: (str) port number of the DB
        :param dbname: (str) database name (creates it if nonexistent)
        :param collection: (str) _collection name (creates it if nonexistent)
        """
        cursor = self.subcollection
        backup_mongo = MongoClient(hostname, portnum)
        backup_db = getattr(backup_mongo, dbname)
        backup_collection = getattr(backup_db, collection)
        backup_collection_string = collection

        print("Beginning storage.")
        for document in cursor:
            backup_collection.insert_one(document)

        print("Ended storage in DB {}".format(backup_collection_string))


class Result(object):
    """
    Stores maximum and minimum result values from a DB object max, min, or avg call.
    """

    def __init__(self, value, collection):
        """

        initialization of Result object

        :param value: the value which is extracted
        :param collection:  (iterable) the collection which the value is a part of
        """

        self.value = value
        self.collection = collection

        if type(value) in dtypes.numbers:
            self.data = [doc for doc in self.collection if doc['y']==self.value]
            self.datum = self.data[0]

