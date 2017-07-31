# TurboWorks
Machine Learning Interface/Black Box Optimizer for FireWorks workflows.


# What is TurboWorks?
TurboWorks is a flexible and easy-to-use automatic machine-learning framework for Fireworks workflows.
### Why would I use it?
If you have a complex, multi-iteration task to execute across different computers, and you would like to automatically reduce the number of expensive calculations needed
to run your task, TurboWorks is for you. 
### What can it do?
TurboWorks functions as a **black box optimizer** for an optimization loop; it requires no knowledge of a function in order to optimize it. More importantly
 though, TurboWorks **retains the workflow management abilties** of FireWorks (provenance, dynamic workflows, duplicate detection and correction, error handling).   
TurboWorks is implemented as a modular, atomic task (FireTask) in a FireWorks workflow; it can run multiple optimizations for a single task or execute
only once in an entire workflow. It's up to you.
 
Other abilities of Turboworks include:
* Facilitating feature engineering
* Duplicate prevention
* Persistent storage and optimization tracking
* Built-in "out-of-the-box" sklearn optimizers
* Support for custom machine learning packages
* Automatic encoding for categorical optimization
* Tuneable control of training and prediction performance, across many kinds of computer resources 

## Requirements
- Python 2 or 3
- NumPy
- SciPy
- Scikit-learn
- FireWorks
- MongoDB

## Installation
~~~~
# Download the repository and install
git clone https://github.com/ardunn/turboworks.git
cd turboworks
pip install . -r requirements

# Make sure Mongo DB's daemon is running
mongod

# Now lets run a minimization example
cd turboworks_examples
python test_basic.py
~~~~

## Tutorial 1: Get Up and running
If you aren't comfortable with Fireworks, please work through the tutorials [here.](https://hackingmaterials.lbl.gov/fireworks/) 

Turboworks is designed for inverse optimization tasks with incremental improvement. For example, a typical workflow might look like this:

Input parameters are given to the first firework. This begins the worklow, and a useful output result is given. The workflow is repeated until enough useful output is obtained (for example, finding a maximum).

Randomly selecting the next input parameters is inefficient, since we will have to execute many workflows. To reduce the required number of computed workflows, we need to intelligently choose new input parameters
with an optimization loop.

This is where Turboworks comes in handy. Turboworks is a Firetask (called `OptTask`) which can go in any firework in the workflow, and which uses `sklearn` regressors to predict the best input parameters,
store them in a MongoDB database, and start a new workflow to compute the next output. The optimization loop then repeats.

The most basic version of `OptTask` implementation requires 4 things:
* **Workflow creator function**: takes in a vector of parameters `x`  and returns a Fireworks workflow. Specified with the `wf_creator` arg to `OptTask`.
* **`'_x_opt'` and `'_y_opt'` fields in spec**: the parameters the workflow is run with and the output metric, in the spec of the Firework containing `OptTask`
* **Dimensions of the search space**: A list of the spaces dimensions, where each dimension is defined by`(higher, lower)` form (for  `float`/ `int`)  or `["a", "comprehensive", "list"]` form for categories. Specified with the `dimensions` argument of `OptTask`
* **MongoDB collection to store data**: Each optimization problem should have it's own collection.


The fastest way to get up and running is to do a simple example. Lets create an optimization loop with one Firework containing two Firetasks, 
`BasicCaclulateTask` and `OptTask`. `BasicCalculateTask` takes in parameters `A`, `B`, and `C` and computes `A*B/C`. We will have `OptTask` run 10 workflows to minimize  `A*B/C` with a `sklearn` `RandomForestRegressor` predictor. 

```
# turboworks_examples/calculate_task.py

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks import FWAction

@explicit_serialize
class BasicCalculateTask(FireTaskBase):
    _fw_name = "BasicCalculateTask"

    def run_task(self, fw_spec):

        A = fw_spec['A']
        B = fw_spec['B']
        C = fw_spec['C']

        score = A*B/C
        return FWAction(update_spec={'_y_opt': score})
```

The following workflow creator function takes in `x`, and returns a workflow based on `x`. Once we start a workflow created with this function, it will keep execute the workflow, predict the next best `x`, and then automatically load another workflow onto the Launchpad using the new `x`. 

```
# turboworks_examples/test_basic.py

from fireworks.core.rocket_launcher import rapidfire
from fireworks import Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from calculate_task import BasicCalculateTask as CalculateTask

def wf_creator(x):

    spec = {'A':x[0], 'B':x[1], 'C':x[2], '_x_opt':x}
    X_dim = [(1, 5), (1, 5), (1, 5)]

    # CalculateTask writes _y_opt field to the spec internally.

    firework1 = Firework([CalculateTask(),
                          OptTask(wf_creator='turboworks_examples.test_basic.wf_creator',
                                  dimensions=X_dim,
                                  host='localhost',
                                  port=27017,
                                  name='turboworks')],
                          spec=spec)

    return Workflow([firework1])
```

`OptTask` must have the `wf_creator` argument specified as a string (similar to `PyTask`). For now we can just specify the MongoDB collection with `host` `port` and `name` as args to `OptTask`, with the collection named `opt_default` by default. The dimensions of the search space are integers from `1-5` in `3` dimensions.

To start the optimization, we run the code below, and we use the point `[5, 5, 2]` as our initial guess.
```
# turboworks_examples/test_basic.py

if __name__ == "__main__":
    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)
    launchpad.reset(password=None, require_password=False)
    launchpad.add_wf(wf_creator([5, 5, 2]))
    rapidfire(launchpad, nlaunches=10, sleep_time=0)
```
We only need to add the workflow created with the workflow creator to the Launchpad one time. Once the first workflow is finished, `wf_creator` is called, and a new workflow is automatically added onto the Launchpad. This is why we can add only one workflow to the Launchpad but launch 10 times.
The output of `$python test_basic.py` is:
```
2017-07-30 22:32:51,204 INFO Performing db tune-up
2017-07-30 22:32:51,213 INFO LaunchPad was RESET.
2017-07-30 22:32:51,216 INFO Added a workflow. id_map: {-1: 1}
2017-07-30 22:32:51,226 INFO Created new dir /Users/alexdunn/TURBOWORKS/turboworks/turboworks_examples/launcher_2017-07-31-05-32-51-226440
2017-07-30 22:32:51,226 INFO Launching Rocket
2017-07-30 22:32:51,821 INFO RUNNING fw_id: 1 in directory: /Users/alexdunn/TURBOWORKS/turboworks/turboworks_examples/launcher_2017-07-31-05-32-51-226440
2017-07-30 22:32:51,824 INFO Task started: {{calculate_task.BasicCalculateTask}}.
2017-07-30 22:32:51,824 INFO Task completed: {{calculate_task.BasicCalculateTask}} 
2017-07-30 22:32:51,824 INFO Task started: {{turboworks.optimize.OptTask}}.
2017-07-30 22:32:52,906 INFO Task completed: {{turboworks.optimize.OptTask}} 
2017-07-30 22:32:52,932 INFO Rocket finished
...
2017-07-30 22:32:54,755 INFO Launching Rocket
2017-07-30 22:32:54,846 INFO RUNNING fw_id: 10 in directory: /Users/alexdunn/TURBOWORKS/turboworks/turboworks_examples/launcher_2017-07-31-05-32-54-755006
2017-07-30 22:32:54,849 INFO Task started: {{turboworks_examples.calculate_task.BasicCalculateTask}}.
2017-07-30 22:32:54,849 INFO Task completed: {{turboworks_examples.calculate_task.BasicCalculateTask}} 
2017-07-30 22:32:54,849 INFO Task started: {{turboworks.optimize.OptTask}}.
2017-07-30 22:32:55,014 INFO Task completed: {{turboworks.optimize.OptTask}} 
2017-07-30 22:32:55,049 INFO Rocket finished
```
Congratulations! We ran our first `OptTask` optimization loop. Now lets take a look at our optimization data we stored in our database.
```
$ mongo
> use turboworks
> db.opt_default.find()
```
* `'x'` indicates the guesses for which expensive workflows have already been run.
* `'x_new'` indicates the predicted best next guess using the available information at that time.
* `'index'` indicates the iteration number (i.e., the number of times the workflow has been run)

The documents containing `lock`, `queue`, and `reserved` are used for duplicate checking for concurrent optimization loops. 

## Tutorial 2: Custom Optimization Algorithms and Extra Features

## Tutorial 3: How to Use `OptTask's` Other Abilities
