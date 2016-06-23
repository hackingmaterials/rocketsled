#This file will hold all classes, methods, etc. for cleaning the DB after
#all workflows have been executed and the FW queue is empty

#THIS FILE IS USELESS AT THE MOMENT

from pymongo import MongoClient

def nuke_collection():
    # Ensure we are referencing the correct DB
    mongo = MongoClient('localhost', 27017)
    db = mongo.TurboWorks
    collection = db.ABC_collection

#Determine number of documents in collection
    cursor = collection.find()
    print '\nNumber of documents: ', cursor.count()

#Delete all data documents, retain all log documents
#This doesnt work for some reason, as if these commands were not here
    # docs_removed = db.collection.delete_many({})
    docs_removed = db.collection.delete_many({'type':'data'})
    num_deleted = docs_removed.deleted_count

    print 'Documents removed:    ', num_deleted
    return num_deleted