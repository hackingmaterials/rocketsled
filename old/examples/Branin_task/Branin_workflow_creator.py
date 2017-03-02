from fireworks import Firework, Workflow

from Branin_task import BraninTask
from old.optimize_task import SKOptimizeTask, DummyOptimizeTask

"""
This file specifies a function that creates a workflow to:
    1. Execute black box function ABCtask
    2. Optimize this function's input parameters
"""

def workflow_creator(input_dict, opt_method):
    """
    :param input_dict (dict): dictionary input of the form:
        {"input": {dictionary of inputs}, "dimensions":{dictionary of 2-tuples defining search space}}
        example:
          input1 = {"A":A, "B":B, "C":C}
          dimensions = {"A_range":(1.0,100.0),"B_range":(1.0,100.0), "C_range":(2,100)}
          input_dict = {'input':input1, 'dimensions':dimensions}

    :param opt_method (str): string defining the optimization method. Options include
        'skopt_gp': gaussian process optimizer (takes integers, float)
        'dummy': random sampling (takes integers, floats)

    :return: wf: a workflow object describing the above workflow using params entered in input_dict
    """

    # Assign FireTasks
    if opt_method=='skopt_gp':
        firetask1 = BraninTask()
        firetask2 = SKOptimizeTask(func='Branin_workflow_creator.workflow_creator', min_or_max="min")
    elif opt_method=='dummy':
        firetask1 = BraninTask()
        firetask2 = DummyOptimizeTask(func='Branin_workflow_creator.workflow_creator')
    else:
        raise ValueError("Invalid optimization method: {}".format(opt_method))

    # Execute FireWork
    fw = [firetask1, firetask2]
    firework1 = Firework(fw, spec=input_dict)
    wf = Workflow([firework1])
    return wf
