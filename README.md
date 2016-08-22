# TurboWorks
Machine Learning Interface/Black Box Optimizer for FireWorks workflows.

# What is TurboWorks?
TurboWorks is a flexible machine-learning framework for maximizing throughput on distributed computing resources.
###Why would I use it?
If you have a complex task to execute, and you would like to use statistical machine learning techniques to reduce the number of expensive calculations needed
to run your task, TurboWorks is for you. 
###What can it do?
TurboWorks functions as a **black box optimizer and task manager**; it requires no knowledge of a function in order to optimize it, and TurboWorks **retains the workflow
management abilties** of FireWorks (provenance, dynamic workflows, duplicate detection and correction, error handling).   
TurboWorks is implemented as a modular atomic task in a FireWorks workflow, a series of complex tasks; it can run multiple optimizations for a single task, or it can execute
only once in an entire workflow.  
###What kind of machine learning does it use?
TurboWorks includes a general purpose **Gaussian process** algorithm (Skopt) which can handle categorical, continuous, discrete integer, and mixed data type optimization tasks.  
TurboWorks also includes a specialized integer-specific Gaussian process library (COMBO) meant to reduce the computational time needed for optimizations on large data sets.  
In addition to these two default optimization algorithms, **the API includes the ability to add user made optimization algorithms** (see 'Adding your own Algorithms' for more).

###Where can I learn more?
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

####Running a basic example in 5 Easy Steps   

__1)__ Navigate to your `TurboWorks` directory.
__2)__ Go to the directory: `examples/Tutorial_integer_example`
__3)__ There are 3 files inside. Lets take a look at `executable.py` first. 
__4)__ Let's run the `graph` function to make sure everything is working correctly. To **reset the fireworks database** and **delete all fireworks data**, enter the day
in YYYY-MM-DD format as `fw_password` argument of `graph`. Set the number of function evaluations `n_runs = 30` parameter of `graph`. For example,


    if __name__=="__main__":
        graph(input_dict, n_runs=30, fw_password='2016-08-16')


__5)__ Now execute this script. The result should be a matplotlib graph showing the best attained score by each algorithm.   

Congrats! Move onto the other tutorials to learn to use TurboWorks for your own problems!

##Tutorial 1: Step by Step guide to using and understanding TurboWorks
An optimization in TurboWorks consists of 3 parts:
* a Firetask which needs to be executed
* a workflow creator function
* a top level script 


Let's say you have a computationally expensive black box function which we call `IntegerTask`:
```
A*B/C = D
```
And you would like to find integer values of `A`, `B`, and `C`, which maximize `D` where `A`, `B`, and `C` range between `1-100`.
TurboWorks will operate with no knowledge of the inner workings of `IntegerTask`. 


__1)__ Navigate to your `TurboWorks` directory.  
__2)__ Go to the directory: `examples/Tutorial_integer_example`  
__3)__ Open `integer_task.py`. This file contains our function in FireTask format. To learn more about how to write a FireTask, see the [FireWorks tutorial page]
(https://pythonhosted.org/FireWorks/guide_to_writing_firetasks.html).  
  
The first portion of code here should remain the same.


    from fireworks.utilities.fw_utilities import explicit_serialize
    from fireworks.core.firework import FireTaskBase, FWAction
    
    @explicit_serialize


We name the class a relevant name, and give it a name in FireWorks as well. 


    class IntegerTask(FireTaskBase):
        _fw_name = "IntegerTask"


The arguments of `run_task` should remain the same as shown here for any FireTask.


    def run_task(self, fw_spec):

In `run_task`, we define our task using the spec `fw_spec`. First, gather inputs:


        # Gather inputs from spec
        A_input = fw_spec['input']['A']
        B_input = fw_spec['input']['B']
        C_input = fw_spec['input']['C']

Now we execute our black box function. 


        # Run black box objective algorithm
        D_output = float(A_input*B_input/C_input)

Finally, we can write the output of the function back to the spec under `output`...


        # Put the calculated output into a dictionary
        D_write = {'output': {'D':D_output}}
        
...And update our spec.


        # Modify spec accordingly
        return FWAction(update_spec=D_write)
        
In summary, we gathered all of the relevant inputs from the fw_spec, calculated an output, and stored the output back in the spec.   
__4)__ Open `integer_task_workflow_creator.py`. This file contains a function which can create an optimization workflow. With this method, we organize how
to execute the black box function and optimize it. In this particular method, we want to compare 3 optimization algorithms; let's look at how
one is assigned.

        def workflow_creator(input_dict, opt_method):

            # Assign FireTasks
            if opt_method=='skopt_gp':
                firetask1 = IntegerTask()
                firetask2 = SKOptimizeTask(func='integer_task_workflow_creator.workflow_creator', min_or_max="max")
                
The first FireTask is our black box task. The second FireTask is the Skopt optimization task.
The optimization task must take the fully defined name of the workflow function as input. We also use the `min_or_max` argument to define
whether we are looking to find a min or max. Now since the FireTasks are assigned, we put them in a workflow.

        # Execute FireWork
        fw = [firetask1, firetask2]
        firework1 = Firework(fw, spec=input_dict)
        wf = Workflow([firework1])
        return wf
        
The dictionary we are assigning to `input_dict` is the dictionary our black box task  in `IntegerTask` will use.  
  
__5)__ Open `executable.py`. This file is the top level script we will run to see how various optimization algorithms perform in this task. We'll skip
all the Matplotlib graphing and just go over the most important parts.
First, we import our other files and the classes we use to execute tasks in Fireworks:  
 
 
    from integer_task_workflow_creator import workflow_creator
    from fireworks.core.rocket_launcher import rapidfire
    from fireworks import FWorker, LaunchPad
    from fireworks.core.rocket_launcher import launch_rocket
    from turboworks.manage_DB import ManageDB
Instantiate a LaunchPad object (for use with FireWorks) and a ManageDB object (for use with TurboWorks database)  

    # Class for managing FireWorks
    launchpad = LaunchPad()

    # Class for managing the TurboWorks database directly
    manageDB = ManageDB()
    
Lets put some sample data into our `input_dict` to start the optimization. We also define the dimensions as a dictionary with each dimension's boudnaries
defined in the format `(upper, lower)`.

    # Sample data
    A = 92
    B = 26
    C = 88
    my_input = {"A":A, "B":B, "C":C}
    dimensions = {"A_range":(1,100),"B_range":(1,100), "C_range":(1,100)}

Now let's put those into our input dictionary:

    # Define the initial input dictionary
    input_dict = {'input':my_input, 'dimensions':dimensions}
    
In our main function `graph`, we define everything we need to run the workflow.
Use FireWorks commands to execute our task. Use TurboWorks `ManageDB` to find the best result we have acquired so far.

    # To reset FireWorks fw_password must be today's date in form 'YYYY-MM-DD'
    launchpad.reset(fw_password, require_password=True)

    gp_best = []
    wf = workflow_creator(input_dict, 'skopt_gp')
    launchpad.add_wf(wf)
    for i in range(n_runs):
        launch_rocket(launchpad)
        gp_best.append(manageDB.get_optima('D', min_or_max='max')[0])
        
The workflow creator function defines the workflow here. Every time it is executed, it returns another workflow, and stores the results
in the default TurboWorks database. During the optimization, the algorithm uses every input in this database.
Finally, we execute the `graph function`:

```
if __name__=="__main__":
    graph(input_dict, n_runs=25, fw_password='2016-08-17')
```


## Tutorial 2: Running your own optimization

##Tutorial 3: Using ManageDB to get useful information from your optimizations
TurboWorks keeps a separate Mongo databse from FireWorks in order to quickly and easily see the inputs and outputs from optimizations. The class that handles several key TurboWorks DB functions is `ManageDB` in `manage_DB.py`
* `__init__`: Specifies the database you'd like to use  
  example: `manageDB = ManageDB(hostname='my_host',portnum=12345, dbname='My_custom_TW_DB', collection='My_custom_TW_collection')`
* `nuke_it`: deletes the entire collection specified during the `ManageDB` object's creation  
  example: `manageDB.nuke_it()`
* `count_it`: counts how many documents are in the collection  
  example: `manageDB.count_it()`
* `query_it`: queries the DB based on typical pymongo syntax  
  example: `x_documents = manageDB.query_it({'x':45}, print_to_console = True)`
* `get_avg`: get the mean of a parameter or output   
  example: `x_average = manageDB.get_avg('x')`
* `get_param`: get all values of the specified param/output as a list  
  example: `x_list = manageDB.get_param('x')`
* `get_optima`: get the maximum/minimum value of a specified param/output  
  example: `x_optimum = manageDB.get_optima('x', 'max')`
* `store_it`  : stores the entire collection in a backup collection  
  example: `manageDB.store_it(hostname='localhost', portnum=27017, dbname='Local_backups', collection='TW_backup)`
   
##Tutorial 4: Implementing your own Optimization Algorithms