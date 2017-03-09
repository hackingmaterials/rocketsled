"""
Utility functions for managing the turboworks databases.

    __init__  : allows custom MongoDB params (ie, store the DB somewhere besides default)
    store     : put custom, ready to go JSON data directly into the turboworks database
    clean     : deletes the entire collection
    count     : counts how many documents are in the collection
    query     : queries the DB based on typical pymongo syntax
    avg       : finds average
    min       : finds min
    max       : finds max
    back_up   : stores the entire collection in a backup collection
"""

from pymongo import MongoClient
from pprint import pprint
import numpy
from references import dtypes
import warnings

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


class DB(object):

    """
    Manages the turboworks databases.

    """

    def __init__(self, hostname='localhost', portnum=27017, dbname='turboworks',
                 collection='turboworks'):
        """
        Allows configuration of the TurboWorks DB

        :param hostname: (str) host name of the MongoDB
        :param portnum: (str) port number of the DB
        :param dbname: (str) database name (creates it if nonexistent)
        :param collection: (str) collection name (creates it if nonexistent)
        """
        self.mongo = MongoClient(hostname, portnum)
        self.db = getattr(self.mongo,dbname)
        self.collection = getattr(self.db,collection)
        self.collection_string = collection

    def store(self, obj):
        """

        A utility for storing turboworks-ready documents in the database.

        :param obj: A dict or list of dicts which can be used with turboworks db.
                for example: obj = {'z':[1,2,3], 'y':5}
        :return: None
        """

        warning_list = "key {} exists in dictionary {}, but is not a list or array. \n" \
                       " this may cause errors internally in turboworks"
        warning_value = "{} must be a {}. Entry skipped. \n" \
                        "Consult turboworks.references.dtypes for a full list of supported datatypes."
        warning_key = "required key {} not in {}. Entry skipped. \n" \
                      " Stored dictionaries must have 'z' and 'y' to be stored in the turboworks db."
        warning_dict = "Object {} is {}, not dict. Entry skipped. \n " \
                       "Turboworks can only store dictionaries in the database."
        error_list = "Object {} is {}, not list. Turboworks can only process a list of dictionaries."

        if isinstance(obj, dict):
            obj = [obj]

        store_dict = None
        if isinstance(obj, list):
            for d in obj:
                if isinstance(d, dict):
                    if 'z' in d.keys():
                        if 'y' in d.keys():
                            if isinstance(d['z'], list) or isinstance(d['z'], numpy.ndarray):
                                if type(d['y']) in dtypes.numbers:
                                    store_dict = {'z': d['z'], 'y': d['y']}
                                    if 'x' in d.keys():
                                        if isinstance(d['x'], list) or isinstance(d['x'], numpy.ndarray):
                                            store_dict['x'] = d['x']
                                        else:
                                            warnings.warn(warning_list.format('x', d))
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

    @property
    def count(self):
        """
        Counts documents in the TurboWorks DB collection (TurboWorks_collection)

        :return: (int) the total number of documents in the collection
        """

        cursor = self.collection.find()
        print('\nNumber of documents:       ', cursor.count())
        return cursor.count()

    @property
    def min(self):
        """
        Finds the minimum value of 'y' in the database.

        :return: (Result object) holds all information pertinent to the minimum 'y' value in the turboworks db.
        """

        Y = [y['y'] for y in self.collection.find()]
        return Result(min(Y), self.collection)

    @property
    def max(self):
        """
        Finds the maximum value of 'y' in the database.

        :return: A Result object holding all information pertinent to the maximum 'y' value in the turboworks db.
        """

        Y = [y['y'] for y in self.collection.find()]
        return Result(max(Y), self.collection)

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
        cursor = self.collection.find(querydict)
        print('Documents matching:        ', cursor.count())
        print('Documents:')

        if print_to_console:
            for document in cursor:
                    pprint(document)

        docs = [document for document in cursor]
        return docs

    def clean(self):
        """
        Deletes all data in the TurboWorks DB collection (TurboWorks_collection)


        :return num_deleted: (int) number of documents deleted from the database
        """
        docs_removed = self.collection.delete_many({})
        num_deleted = docs_removed.deleted_count
        try:
            print 'Documents removed:         ', num_deleted
        except:
            print('Documents removed:         ', num_deleted)
        print('Documents remaining:        0')
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

        X = [x[var] for x in self.collection.find()]

        mean = numpy.mean(X, 0).tolist()
        return mean

    @classmethod
    def reset(self):
        """
        Resets the meta and turboworks databases by deleting all their data.

        :return: None
        """

        db = DB()
        db_meta = DB(collection='meta')
        db.clean()
        db_meta.clean()


    def back_up(self, hostname='localhost', portnum=27017, dbname='turboworks',
                collection='backup'):
        """
        Transfers all data from the TurboWorks DB to another DB

        :param hostname: (str) host name of the MongoDB
        :param portnum: (str) port number of the DB
        :param dbname: (str) database name (creates it if nonexistent)
        :param collection: (str) collection name (creates it if nonexistent)
        """
        cursor = self.collection.find()
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
            self.data = [x for x in self.collection.find({'y':value})]
            self.datum = self.data[0]

