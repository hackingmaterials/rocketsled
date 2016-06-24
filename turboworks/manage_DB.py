#This file will hold all functions for cleaning the DB after
#all workflows have been executed and the FW queue is empty
from pymongo import MongoClient

#Set up DB
mongo = MongoClient('localhost', 27017)
db = mongo.TurboWorks
collection = db.ABC_collection

def nukeit():
#Delete all documents
#Ensure we are referencing the correct DB
    docs_removed = collection.delete_many({})
    num_deleted = docs_removed.deleted_count
    print 'Documents removed:   ', num_deleted
    print 'Documents remaining:  0'
    return num_deleted

def countit():
#Determine number of documents in collection
    cursor = collection.find()
    print '\nNumber of documents: ', cursor.count()
    return cursor.count()