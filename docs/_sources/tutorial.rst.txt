=============================================================
Welcome to the :code:`rocketsled` tutorial! - 30 min required
=============================================================

This tutorial is based on the example file found in rocketsled/examples/basic.py.


What you'll need
----------------

1. An expensive objective function (or Fireworks workflow, if you already have it)
2. The search domain of your objective function.
3. A working knowledge of FireWorks (see the `FireWorks <https://github.com/materialsproject/fireworks>`_ docs for a quick refresher, if needed!)


A bird's eye view:
-----------------------------------
This tutorial will walk you through setting up an optimization on your local machine. For more advanced execution options, see the FireWorks documentation and the :doc:`Comprehensive Guide to rocketsled </guide>`.


0. **Set up prerequisites** Getting mongodb running, and FireWorks and rocketsled installed.

1. **Create a workflow creator containing your objective function and OptTask (optimization task).** Your objective function should take in a list of parameters x, and return a scalar (or list of scalars, if multiobjecive), y. Your workflow should be set up so that the parameters x and y get written to the spec of the Firework containing OptTask, using the keys "_x" and "_y" - if you don't know how to do this, don't worry: we'll walk you through it.

2. **Configure the optimization with rocketsled MissionControl.** MissionControl is the tool rocketsled provides to easily configure the optimization and execution parameters.

3. **Run your optimization** (using FireWorks' LaunchPad).


0. Setting up prerequisites
---------------------------

First, pip install rocketsled:

.. code-block:: bash

    $ pip install rocketsled


This pip install should automatically pip install FireWorks. If it does not, make sure to:

.. code-block:: bash

    $ pip install FireWorks

Last, make sure you have a mongodb instance running locally.

.. code-block:: bash

    $ mongod

Next, find your rocketsled directory and ``cd`` to the directory we will work through.

.. code-block:: bash

    $ cd ./rocketsled/examples

This example is based on the basic.py file.


1. Create a workflow creator containing your objective function and OptTask (optimization task).
------------------------------------------------------------------------------------------------

*Note: As you read along, try modifying and running the correspondigg example file in rocketsled/examples/basic.py.*

Your objective function may be a Fireworks workflow already, or it may just be
a python (or other language) function or script. Whatever the case, the best
way to use rocketsled is with a **workflow creator function**. This function should
take in an input x vector and return a workflow which determines the output y
scalar (or vector, if multiobjective).

Rocketsled has a few requirements for the kinds of workflow creator functions
which are valid:

    1. The optimization task OptTask is passed into a FireWork in the workflow.
    2. The fields "_x" and "_y" are written to the spec of the FireWork containing OptTask. Here, "_x" is the input and "_y" is the workflow output.
    3. You use MissionControl's "configure" method to set up the optimization, and pass in wf_creator as it's first argument.

We'll take care of the first two requirements now.

First, let's set up our LaunchPad and ``opt_label``. The ``opt_label`` defines
where the optimization data is stored, in the same database as the FireWorks LaunchPad.

.. code-block:: python

    # Setting up the FireWorks LaunchPad
    launchpad = LaunchPad(name='rsled', host='localhost', port=27017)
    opt_label = "opt_default"
    db_info = {"launchpad": launchpad, "opt_label": opt_label}


If you already have an objective function as a python function, such as the
*very* simplified one below:

.. code-block:: python

    def obj_function(x):
        y = x[0] * x[1] / x[2]
        return y


We can put this function into a workflow by writing it as a custom FireTask.

.. code-block:: python

    from fireworks.core.firework import FireTaskBase
    from fireworks.utilities.fw_utilities import explicit_serialize

    @explicit_serialize                              # tells FireWorks where to find this task
    class ObjectiveFuncTask(FireTaskBase):           # Make sure our custom FireTask inherits from FireTaskBase, the parent class for custom FireTasks
        _fw_name = "ObjectiveFuncTask"               # A nice name for our Firework
        def run_task(self, fw_spec):                 # run_task is a required method for every FireTask
            x = fw_spec['_x']                        # Read the input vector from the Fireworks spec
            y = x[0] * x[1] / x[2]                   # Compute the output value (trivial in our example case)
            return FWAction(update_spec={'_y': y})   # FWAction is an object that modifies the workflow; here, update_spec adds the y output to the containing Firework's spec.

*Note that we write the "_y" field to the spec; this is required by rocketsled!*

Now that we have the objective function as a FireTask, we can easily create a workflow in FireWorks. In this workflow,
we'll just use a single Firework with two sequential FireTasks. The first Firetask evaluates your objective function, and
the second evaluates the optimization.

.. code-block:: python

    def wf_creator(x):
        spec = {'_x': x}
        firework1 = Firework([ObjectiveFuncTask(), OptTask(**db_info)], spec=spec)
        return Workflow([firework1])

Let's also define some constraints for our objective function, and put them in a variable called ``x_dim``. We'll use this later.
The dimensions are defined in a list of lists/tuples, with one list/tuple for each dimension. To use a range of floats or ints, pass in a 2-tuple in the form ``(lower, higher)``. To pass in categorical variables or sets of discontinuous ints/floats, pass in lists.
Let's constrain this problem to integers between 1 and 5 (inclusive) in each of the three dimensions.

.. code-block:: python

    # We constrain our dimensions to 3 integers, each between 1 and 5
    x_dim = [(1, 5), (1, 5), (1, 5)]

Alternatively, we could define our dimensions with some discrete entries:

.. code-block:: python

    # Each list dimension has the explicit allowed points, while 2-tuples are ranges
    x_dim = [[1.98, 2.99, 3.45, 1.09, 199.4], (1.0, 100.0), (1, 20)]

These dimensions allow 5 possible floats for x[0], all floats between 1 and 100 for x[1], and all ints between 1 and 20 (inclusive) for x[2].

Furthermore, we could even define our dimensions with all discrete entries:

.. code-block:: python

    # Each list dimension has the explicit allowed points
    x_dim = [[1.98, 2.99, 3.45, 1.09, 199.4], [100.928, 98.38, 97.45, 45.32, 23.99], [1, 19, 25, 63, 18]]


These dimensions allow 5 possible floats for x[0], 5 other possible floats for x[1], and 5 integers for x[2].
*To use categorical dimensions, simply pass in a list of strings for a dimension; see the complex.py example for an example.*

Great! Our workflow creator function is now set up and ready to go. If you have an objective function workflow
with more complexity than a single FireTask can handle, simply change the above wf_creator for your workflow,
placing the OptTask in the same FireWork where your final result(s) are calculated. Again **make sure you have "_x" and "_y" fields in the spec of the FireWork OptTask is in!**.
See complex.py example for an example of a more complex workflow (and optimization).


2. Configure the optimization with rocketsled's MissionControl.
---------------------------------------------------------------

``MissionControl`` is the way optimization configuration is done in rocketsled. First, we make a ``MissionControl`` object with
the database info. After this is done, we can configure the optimization and start (launch) our optimization!

.. code-block:: python

    if __name__ == "__main__":
        # Make a MissionControl object
        mc = MissionControl(**db_info)

        # Reset the launchpad and optimization db for this example
        launchpad.reset(password=None, require_password=False)
        # the MissionControl reset simply gets rid of any optimization data left over from previous runs of this example
        mc.reset(hard=True)

        # Configure the optimization db with MissionControl
        mc.configure(wf_creator=wf_creator, dimensions=x_dim)


The ``configure`` method defines all the parameters for optimization. We can change optimization algorithms, define external optimization algorithms, change optimization parameters (e.g., number of search points), change parallelism parameters, and much more.
But by default, all we need to do is pass in the wf_creator and the dimensions of the problem.


3. Run the optimization.
------------------------

Now, we are able to launch our optimization.

.. code-block:: python

        # Run the optimization loop 10 times.
        launchpad.add_wf(wf_creator([5, 5, 2]))              # add a workflow to the LaunchPad
        rapidfire(launchpad, nlaunches=10, sleep_time=0)     # Launch 10 workflows

If everything is working right, you should see the log output from the optimization.

.. code-block:: bash

    2018-12-31 18:05:11,416 INFO Performing db tune-up
    2018-12-31 18:05:11,821 INFO LaunchPad was RESET.
    2018-12-31 18:05:11,822 INFO Optimization collection opt_default hard reset.
    2018-12-31 18:05:11,876 INFO Rocketsled configuration succeeded.
    2018-12-31 18:05:11,891 INFO Added a workflow. id_map: {-1: 1}
    2018-12-31 18:05:11,912 INFO Created new dir /Users/ardunn/alex/lbl/projects/rocketsled/code/rocketsled/rocketsled/examples/launcher_2019-01-01-02-05-11-911794
    2018-12-31 18:05:11,912 INFO Launching Rocket
    2018-12-31 18:05:11,954 INFO RUNNING fw_id: 1 in directory: /Users/ardunn/alex/lbl/projects/rocketsled/code/rocketsled/rocketsled/examples/launcher_2019-01-01-02-05-11-911794
    2018-12-31 18:05:11,962 INFO Task started: {{basic.BasicTask}}.
    2018-12-31 18:05:11,962 INFO Task completed: {{basic.BasicTask}}
    ...


We can also use MissionControl to track our optimization separately from the execution of the workflows.

.. code-block:: python

    # Examine results
    plt = mc.plot()
    plt.show()

The output summary should appear:

.. code-block:: bash

    Optimization Analysis:
    Number of objectives: 1
        Number of optima: 1
            min(f(x))  is 0.2 at x = [1, 1, 5]

    Problem dimension:
        * X dimensions (3): [<class 'int'>, <class 'int'>, <class 'int'>]
        * Z dimensions (0): []

    Number of Optimizations: 10
    Optimizers used (by percentage of optimizations):
        * 100.00%: RandomForestRegressor with acquisition: Expected Improvement
    Number of reserved guesses: 1
    Number of waiting optimizations: 0
    DB not locked by any process (no current optimization).

And the optimization plot should appear similar to:

    .. image:: _static/opt_basic.png
       :alt: server
       :align: center
       :width: 1000px

Great! This concludes the tutorial. Please see the rocketsled/examples/complex.py example, the :doc:`Comprehensive Guide to rocketsled </guide>`, or the `FireWorks documentation <https://github.com/materialsproject/fireworks>`_ for more details.


