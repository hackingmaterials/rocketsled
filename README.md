# TurboWorks
Machine Learning Interface for FireWorks workflows

To learn more about FireWorks, see the [official documentation] (https://pythonhosted.org/FireWorks/)  
To learn more about the optimization algorithm, Skopt, see the [official GitHub Repo] (https://github.com/scikit-optimize/scikit-optimize)

### Software Requirements:
- Python 2 or 3
- NumPy
- SciPy
- Scikit-Learn
- Scikit-Optimize
- FireWorks
- MongoDB

### Installation
```
git clone https://github.com/ardunn/TurboWorks
cd TurboWorks
pip install -r requirements.txt
python setup.py develop
```

You will also need to install Skopt:
```
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize
pip install -r requirements.txt
python setup.py develop
```

### Get up and running
Have a complex workflow you need to execute? [Read more about how to use FireWorks.] (https://pythonhosted.org/FireWorks/)

**TurboWorks is implemented as a single [FireTask] (https://pythonhosted.org/FireWorks/guide_to_writing_firetasks.html) in Fireworks**  
It will be executed as part of a FireWork, which is part of a workflow.  

Have a mongod instance running [via MongoDB, read more about MongoDB here](https://docs.mongodb.com/getting-started/shell/)

*coming soon*

### Implementing your own Optimization Algorithms
*coming soon*