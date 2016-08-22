# TurboWorks
Machine Learning Interface/Black Box Optimizer for FireWorks workflows.

# What is TurboWorks?
TurboWorks is a flexible machine-learning framework for maximizing throughput on distributed computing resources.
####Why would I use it?
If you have a complex task to execute, and you would like to use statistical machine learning techniques to reduce the number of expensive calculations needed
to run your task, TurboWorks is for you. 
####What can it do?
TurboWorks functions as a **black box optimizer and task manager**; it requires no knowledge of a function in order to optimize it, and TurboWorks **retains the workflow
management abilties** of FireWorks (provenance, dynamic workflows, duplicate detection and correction, error handling).   
TurboWorks is implemented as a modular atomic task in a FireWorks workflow, a series of complex tasks; it can run multiple optimizations for a single task, or it can execute
only once in an entire workflow.  
####What kind of machine learning does it use?
TurboWorks includes a general purpose **Gaussian process** algorithm (Skopt) which can handle categorical, continuous, discrete integer, and mixed data type optimization tasks.  
TurboWorks also includes a specialized integer-specific Gaussian process library (COMBO) meant to reduce the computational time needed for optimizations on large data sets.  
In addition to these two default optimization algorithms, **the API includes the ability to add user made optimization algorithms** (see 'Adding your own Algorithms' for more).

####Where can I learn more?
To learn more about FireWorks, see the [official documentation] (https://pythonhosted.org/FireWorks/)  
To learn more about the general default optimization algorithm, Skopt, see the [official GitHub Repo] (https://github.com/scikit-optimize/scikit-optimize)  
To learn more about the specialized integer optimization algorithm, see the [official GitHub Repo] (https://github.com/tsudalab/combo)

## Software Requirements:
- Python 2 or 3
- NumPy
- SciPy
- Scikit-Learn
- FireWorks
- MongoDB
- Optimization APIs (Scikit-Optimize, Tsudalab COMBO)

## Installation

To install the guaranteed working (as of 8/20/2016) version of TurboWorks, run:
```
git clone https://github.com/ardunn/TurboWorks
cd TurboWorks
pip install -r requirements_lite.txt
python setup.py develop
cd optimizers
cd combo
python setup.py install
cd .. 
cd scikit-optimize
python  setup.py install
```

**OR**  
To install using the latest versions of COMBO and Skopt, run:
```
git clone https://github.com/ardunn/TurboWorks
cd TurboWorks
pip install -r requirements_full.txt
python setup.py develop
```
This installation will download the lastest optimization suites of COMBO and Skopt, as well as all their requirements.
Since both suites are under ongoing development, they may not work as intended when downloaded via the requirements.


## Get up and running

####Get familiar with FireWorks
Have a complex scientific workflow you need to execute? Don't know about FireWorks? [Read more about how to use FireWorks.] (https://pythonhosted.org/FireWorks/)
**TurboWorks is implemented as a single [FireTask] (https://pythonhosted.org/FireWorks/guide_to_writing_firetasks.html) in Fireworks**  
It will be executed as part of a FireWork, which is part of a workflow.    

####Get familiar with MongoDB
Don't know about MongoDB? [read about and download MongoDB here] (https://docs.mongodb.com/getting-started/shell/)  
Once you have a MongoDB database set up, start the `mongod` process by typing `mongod` into a terminal, or, if your database is stored somewhere
besides the default db location, use `mongod --dbpath path/to/my/database/data`

####Running a basic example

You have a computationally expensive black box function which we call `IntegerTask`:
```
A*B/C = D
```
And you would like to find integer values of `A`, `B`, and `C`, which maximize `D` where `A`, `B`, and `C` range between `1-100`.   

1. Navigate to your `TurboWorks` directory.
2. Go to the directory: `examples/Tutorial_integer_example`
3. The files inside are:
  * `integer_task.py`: Your black box function as a FireTask which accepts 3 integer inputs (`A,B,C`) and returns a float (`D`). 
  * `integer_task_workflow_creator.py`: a function for creating and returning a workflow which executes `IntegerTask` and optimizes for the next trial. 
  * `executable.py`: the top level executable script which will make a graph demonstrating the effect of optimization over time.
4. Lets take a look at `executable.py` first. 
5. Let's run the `graph` function to make sure everything is working correctly. To **reset the fireworks database** and **delete all fireworks data**, enter the day
in YYYY-MM-DD format as `fw_password` argument of `graph`. Set the number of function evaluations using the `n_runs` parameter of `graph`. For example,
```
if __name__=="__main__":
    graph(input_dict, n_runs=30, fw_password='2016-08-16')
```
6. Now execute this script. The result should be a matplotlib graph showing the best attained score by each algorithm.
6. Congrats! Move onto the other tutorials to learn to use TurboWorks for your own problems!

##Tutorials

####Step by Step guide to using and understanding TurboWorks

####Using ManageDB utility methods to get useful information from your optimizations
TurboWorks keeps a separate Mongo databse from FireWorks in order to quickly and easily see the inputs and outputs from optimizations.  
The class that handles several key TurboWorks DB functions is `ManageDB` in `manage_DB.py`
* `__init__`: Gives options for instantiating the database methods  
  example: `manageDB = ManageDB(hostname='my_host',portnum=12345, dbname='My_custom_TW_DB', collection='My_custom_TW_collection')`
* `nuke_it`: deletes the entire collection specified during the `ManageDB` object's creation  
  example: `manageDB.nukeit()`
* `count_it`: counts how many documents are in the collection  
  example: `manageDB.countit()`
* `query_it`: queries the DB based on typical pymongo syntax  
  example: `matching_docs = manageDB.queryit({'myvar':45}, print_to_console = True)`
* `get_avg`: get the mean of a parameter 
* `get_param`: get all values of the specified param/output
* `get_optima`: get the maximum/minimum value of a specified param/output
* `store_it`  : stores the entire collection in a backup collection
   

#### Implementing your own Optimization Algorithms
*coming soon*