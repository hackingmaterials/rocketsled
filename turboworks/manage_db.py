from pymongo import MongoClient
from pprint import pprint
import numpy as np

"""
Utility functions for managing the turboworks database.

    __init__  : allows custom MongoDB params (ie, store the DB somewhere besides default)
    nuke      : deletes the entire collection
    count     : counts how many documents are in the collection
    query     : queries the DB based on typical pymongo syntax
    get_avg   : get the mean of a set of params or output
    get_optima: get the maximum/minimum value of a specified param/output
    back_up   : stores the entire collection in a backup collection
"""

import numpy

class ManageDB():

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

    def nuke(self):
        """
        deletes all data in the TurboWorks DB collection (TurboWorks_collection)
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

    def count(self):
        """
        counts documents in the TurboWorks DB collection (TurboWorks_collection)
        :return cursor.count(): (int) the total number of documents in the collection
        """
        cursor = self.collection.find()
        print('\nNumber of documents:       ', cursor.count())
        return cursor.count()

    def query(self, querydict=None, print_to_console = False):
        """
        queries documents via PyMongo's find()

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

    def avg(self, var):
        """
        :param var: (string) the variable to be averaged. Should be 'Z', or 'X' or 'Y'
            example: get_avg("Z")
        :return mean: (list) of average values across the db for each of the features in var
            example: db has two docs, {'X':[1,2,3]} and {'X':[2,4,6]}. Returns [1.5, 3, 4.5].
        """

        if var in ['X', 'Y', 'Z']:
            var = var.lower()


        X = [x[var] for x in self.collection.find()]

        try:
            num_features = len(X[0])
            list_by_feature = []
            for i in range(num_features):
                list_by_feature.append([x[i] for x in X])
        except TypeError:
            list_by_feature = X

        mean = numpy.mean(list_by_feature, 0).tolist()
        return mean

    def min(self):
        """
        :return: The minimum of 'y', the output scalar.
        """
        Y = [y['y'] for y in self.collection.find()]
        return min(Y)

    def max(self):
        """
        :return: The maximum of 'y', the output scalar.
        """
        Y = [y['y'] for y in self.collection.find()]
        return max(Y)

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
