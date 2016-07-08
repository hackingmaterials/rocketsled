# TurboWorks
Machine Learning Interface for FireWorks workflows

To learn more about FireWorks, see the [official documentation] (https://pythonhosted.org/FireWorks/)

Software Requirements:
Python 3.X
NumPy
SciPy
Scikit-Learn
Scikit-Optimize
FireWorks

File list:
__init__.py
test_code.py            [primarily runs code]
ABC_task.py             [arbitrary black box task]
ABC_workflow_creator.py [creates a workflow with black box task and optimizer]
dummy_opt.py            [dummy (random) optimization algorithm]
gp_opt.py               [modified Skopt file, replaces gp_opt.py in scikit-optimize/scikit-optimize/skopt]
manage_DB.py            [mongoDB utility collection for checking algorithm effectiveness]
optimize_task.py        [linking FireWorks and Scikit-Optimize
