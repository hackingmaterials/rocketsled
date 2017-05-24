# TurboWorks
Machine Learning Interface/Black Box Optimizer for FireWorks workflows.


# What is TurboWorks?
TurboWorks is a flexible and easy-to-use automatic machine-learning framework for Fireworks workflows.
### Why would I use it?
If you have a complex task to execute across different computers, and you would like to use machine learning techniques to reduce the number of expensive calculations needed
to run your task, TurboWorks is for you. 
### What can it do?
TurboWorks functions as a **black box optimizer** for an optimization loop; it requires no knowledge of a function in order to optimize it. More importantly
 though, TurboWorks **retains the workflow management abilties** of FireWorks (provenance, dynamic workflows, duplicate detection and correction, error handling).   
TurboWorks is implemented as a modular, atomic task (FireTask) in a FireWorks workflow; it can run multiple optimizations for a single task, or it can execute
only once in an entire workflow. It's up to you.


### Where can I learn more?
To learn more about FireWorks, see the [official documentation] (https://pythonhosted.org/FireWorks/)  

## Requirements:
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

Let's take a look at how Turboworks works. 

Turboworks is a FireTask inside Fireworks called `OptTask`.
`OptTask` stores all the optimization data from your Firework or workflow inside its own Mongodb collection.
Using the data in this collection, it uses scikit learn regressors (or your own algorithms) to recommend the next best guess.



## Tutorial 2: Implementing your own Optimization Algorithms
