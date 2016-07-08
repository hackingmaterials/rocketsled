#This file will hold all functions for cleaning the DB after
#all workflows have been executed and the FW queue is empty
from pymongo import MongoClient
from pprint import pprint
import numpy as np

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
    print ('Documents removed:         ', num_deleted)
    print ('Documents remaining:        0')
    return num_deleted


def count_it():
    """
	counts documents in the TurboWorks DB collection (TurboWorks_collection)
	"""
    cursor = collection.find()
    print ('\nNumber of documents:       ', cursor.count())
    return cursor.count()


def query_it(querydict=None):
	"""
	queries documents in the TurboWorks DB collection (TurboWorks_collection)
	:param querydict (dict): a dictionary query entry compatible with pymongo syntax
	"""
	if querydict is None:
		querydict={}
	cursor = collection.find(querydict)
	print ('Documents matching:        ', cursor.count())
	print ('Documents:')
	for document in cursor:
		pprint(document)


def get_optima(output_var, min_or_max='min'):
	"""
	:param output_var: a string representing the variable you want the optimum for
	:param min_or_max: 'min' finds minimum (default), 'max' finds maximum
	:return: maximum/minimum value
	"""
	min = None
	max = None
	optima_params ={}
	cursor = collection.find()
	for document in cursor:
		for key in document:
			if key == output_var:
				if min_or_max == 'min':
					if min!=None:
						if document[key]<min:
							min = document[key]
							optima_params = document
					elif min==None:
						min = document[key]
						optima_params = document
					else:
						print('Manage_DB: Incorrect datatype')
				elif min_or_max == 'max':
					if max!=None:
						if document[key]>= max:
							max = document[key]
							optima_params = document
					elif max==None:
							max = document[key]
							optima_params = document
					else:
						print('Manage_DB: Incorrect datatype')
				else:
					print("Invalid option for min_or_max \nUsing minimum")
					get_optima(output_var)

	if min_or_max=='max':
		return max
	elif min_or_max=='min':
		return min
	else:
		print("Invalid option for min_or_max \nUsing minimum")
		get_optima(output_var)


def get_avg(var):
	total_list = []
	cursor = collection.find()
	for document in cursor:
		for key in document:
			if key == var:
				total_list.append(document[key])
	average=0
	if type(total_list[0]) == np.int64 or type(total_list[0]) == int:
		average = int(sum(total_list)/len(total_list))
	elif type(total_list[0])== np.float64 or type(total_list[0])==float:
		average = float(sum(total_list)/len(total_list))
	return average

def get_param(var):
	total_list = []
	cursor = collection.find()
	for document in cursor:
		for key in document:
			if key == var:
				total_list.append(document[key])
	return total_list