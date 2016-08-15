# TurboWorks
Machine Learning Interface/Black Box Optimizer for FireWorks workflows.

# What is TurboWorks?
TurboWorks is a flexible machine-learning framework for maximizing throughput on distributed computing resources.
####Why would I use it?
If you have a complex task to execute, and you would like to use statistical machine learning techniques to reduce the number of expensive calculations needed
to run your task, TurboWorks is for you. 
####What can it do?
TurboWorks functions as a black box optimizer and task manager; it requires no knowledge of a function in order to optimize it, and TurboWorks retains the workflow
management abilties of FireWorks (provenance, dynamic workflows, duplicate detection and correction, error handling). 
TurboWorks is implemented as a modular atomic task in a FireWorks workflow, a series of complex tasks; it can run multiple optimizations for a single task, or it can execute
only once in an entire workflow.
####What kind of machine learning does it use?
TurboWorks includes a general purpose gaussian process algorithm (Skopt) which can handle categorical, continuous, discrete integer, and mixed data type optimization tasks.
TurboWorks also includes a specialized integer-specific gaussian process library (COMBO) meant to reduce the computational time needed for optimizations on large data sets. 
In addition to these two default optimization algorithms, the API includes the ability to add user made optimization algorithms (see 'Adding your own Algorithms' for more).  

####Where can I learn more?
To learn more about FireWorks, see the [official documentation] (https://pythonhosted.org/FireWorks/)  
To learn more about the general default optimization algorithm, Skopt, see the [official GitHub Repo] (https://github.com/scikit-optimize/scikit-optimize)
To learn more about the specialized integer optimization algorithm, 

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
Have a complex workflow you need to execute? [Read more about how to use FireWorks.] (https://pythonhosted.org/FireWorks/)

**TurboWorks is implemented as a single [FireTask] (https://pythonhosted.org/FireWorks/guide_to_writing_firetasks.html) in Fireworks**  
It will be executed as part of a FireWork, which is part of a workflow.  

Have a mongod instance running [via MongoDB, read more about MongoDB here](https://docs.mongodb.com/getting-started/shell/)

*coming soon*

### Implementing your own Optimization Algorithms
*coming soon*