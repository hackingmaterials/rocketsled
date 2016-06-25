import importlib
#testing concepts to see why optimize_task is having importing problem
class someClass():
	def __init__(self,name):
		x = importlib.import_module(name)
		print('the module imported')
