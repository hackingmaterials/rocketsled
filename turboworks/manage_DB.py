#This file will hold all functions for cleaning the DB after
#all workflows have been executed and the FW queue is empty
from pymongo import MongoClient
from pprint import pprint

#Set up DB
mongo = MongoClient('localhost', 27017)
db = mongo.TurboWorks
collection = db.TurboWorks_collection

def nuke_it():
    """
    deletes all data in the TurboWorks DB collection (TurboWorks_collection)
    """
    docs_removed = collection.delete_many({})
    num_deleted = docs_removed.deleted_count
    print 'Documents removed:         ', num_deleted
    print 'Documents remaining:        0'
    return num_deleted

def count_it():
    """
	counts documents in the TurboWorks DB collection (TurboWorks_collection)
	"""
    cursor = collection.find()
    print '\nNumber of documents:       ', cursor.count()
    return cursor.count()

def query_it(querydict=None):
	"""
	queries documents in the TurboWorks DB collection (TurboWorks_collection)
	"""
	if querydict is None:
		querydict={}
	cursor = collection.find(querydict)
	print 'Documents matching:        ', cursor.count()
	print 'Documents:'
	for document in cursor:
		pprint(document)
