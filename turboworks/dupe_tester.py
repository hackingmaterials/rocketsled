from pymongo import MongoClient


mongo = MongoClient(host='localhost', port=27017)
tw = mongo.turboworks.turboworks
meta = mongo.turboworks.meta


def find_dupes(db):
    dupes = []
    og = []
    for doc in db.find():
        if doc['z'] not in og:
            og.append(doc['z'])
        else:
            dupes.append(doc['z'])


    print len(dupes)
    print dupes


find_dupes(tw)
find_dupes(meta)