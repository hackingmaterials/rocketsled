*Note: This quickstart assumes a limited knowledge of FireWorks. If you already have a workflow built, see the examples or the more advanced tutorials.*


=====================================================
Welcome to the :code:`rocketsled` quickstart! - 5 min
=====================================================


If you have a Python function to optimize, the easiest way to get started is to use rocketsled's auto_setup. Auto-setup wraps any Python function in a FireWork - an execution wrapper -, creates a Firework containing an OptTask optimization, and creates a workflow optimization loop linking the two Fireworks which is ready for launch.

Let's get an optimization running on your local machine. First, make sure a :code:`mongod` instance is running.

.. code-block:: bash

    $ mongod


Define objective function
-------------------------

Great! Now lets define a trivial objective function f(x) for this demo. Your actual objective function will be **much** more complex than this.

.. code-block:: python

    # The objective function must accept a vector and return a scalar/vector.
    def f(x):
        return x[0] * x[1] / x[2]

Define constraints
------------------

Let's constrain this function in each of its dimensions. With rocketsled, each bounded dimension is represented as a 2-tuple for (low, high), and is placed in a list. So if we want to constrain x :sub:`0` to integers between 1-100, x :sub:`1` to integers between 200-300, and x :sub:`3` to floats between 5.0-10.0:

.. code-block:: python

    dimensions = [(1, 100), (200, 300), (5.0, 10.0)]


These constraints ensure the function has a maximum value of 6,000.


Using auto_setup on a function
--------------------------------

Now we can use :code:`auto_setup` to write a file containing

   1. A workflow creator that can:

      a. Run your function in a FireWork

      b. Run the optimization algorithm in a separate FireWork

   2. Commands to launch your workflow.


Lets' maximize our objective function using rocketsled's default predictor, based on scikit-learn's RandomForestRegressor.

.. code-block:: python

    from rocketsled import auto_setup

    # Define the db where the LaunchPad and optimization data will be stored
    # The 'opt_label' field defines the name of the optimization collection
    dbinfo = {"host": "localhost", "name": "my_db", "port": 27017, "opt_label": "quickstart"}

    if __name__ == "__main__":
        auto_setup(f, dimensions, wfname="quickstart", maximize=True, **dbinfo)

.. code-block:: bash

    File successfully created!
    Find your auto sled at /Users/home/rocketsled/rocketsled/auto_sleds/quickstart.py


Check out and run the auto sled
--------------------------------

Let's go to this directory and look at the file, which should look similar to this:

.. code-block:: python

    from __future__ import unicode_literals
    """
    This is an automatically created script from auto_setup.
    If you are not comfortable working with FireWorks and PyTask, do NOT move this
    file out this directory or rename it if you want to run this workflow!

    If you are comfortable working with FireWorks and PyTask, feel free to edit
    and/or move this file to suit your needs. See the OptTask documentation and the
    examples for more information on setting up workflow creators.
    """
    from fireworks import PyTask, Firework, Workflow, LaunchPad
    from fireworks.core.rocket_launcher import rapidfire
    from rocketsled.utils import deserialize, random_guess
    from rocketsled import OptTask


    # This is your function, imported to rocketsled to use with PyTask.
    f = deserialize('/Users/ardunn/quickstart.f')

    def wf_creator(x):
        spec = {'_x_opt':x}
        pt = PyTask(func='rocketsled.auto_sleds.quickstart.f', args=[x], outputs=['_y_opt'])
        ot = OptTask(opt_label='quickstart', dimensions=[(1, 100), (200, 300), (5.0, 10.0)], wf_creator='rocketsled.auto_sleds.quickstart.wf_creator', maximize=True, host='localhost', port=27017, name='my_db')
        fw0 = Firework([pt], spec=spec, name='PyTaskFW')
        fw1 = Firework([ot], spec=spec, name='RocketsledFW')
        wf = Workflow([fw0, fw1], {fw0: [fw1], fw1: []}, name='quickstart @ ' + str(x))
        return wf


    if __name__=='__main__':

        # Make sure the launchpad below is correct, and make changes if necessary if
        # it does not match the OptTask db ^^^:
        lpad = LaunchPad(host='localhost', port=27017, name='my_db')
        # lpad.reset(password=None, require_password=False)

        # Define your workflow to start...
        wf1 = wf_creator(random_guess([(1, 100), (200, 300), (5.0, 10.0)]))

        # Add it to the launchpad and launch!
        lpad.add_wf(wf1)
        # rapidfire(lpad, nlaunches=5, sleep_time=0)

:code:`wf_creator` returns an optimization loop Workflow containing your objective function Firework and the optimization Firework. Then it adds it to the launchpad and launches it!

Your workflow on the launchpad looks like this:

.. image:: _static/quickstart_lp.png
   :alt: quickstart_viz
   :width: 1200px

Your objective function is contained in PyTaskFW. The optimization is done in RocketsledFW. When both Fireworks have completed, RocketsledFW launches another workflow based on the next best predicted x value.

Uncomment the :code:`lpad.reset` line if necessary (i.e., if this database is not already a FireWorks db or you don't mind resetting it). Uncomment the last line if you'd like to launch right away! Let's change nlaunches to 100, to run the first 100 Fireworks (50 optimization loops).

.. code-block:: python

    rapidfire(lpad, nlaunches=100, sleep_time=0)



Visualize the optimization results
----------------------------------

Rocketsled comes with a simple function for creating a matplotlib optimization plot.

.. code-block:: python

    from rocketsled import visualize
    from fireworks import LaunchPad

    lp = LaunchPad(host='localhost', port=27017, name='my_db')
    visualize(lp.db.quickstart, maximize=True)

.. image:: _static/quickstart_viz1.png
   :alt: quickstart_viz
   :width: 1200px

The best found value is shown in green.
Although for this basic example we are using relatively few search points (default 1,000) and no acquisition function for the Bayesian optimization (acq=None, default), you should still find that the maximum found is 90-99% of the true maximum, 6,000.


Congrats! We've just worked through the deployment and execution of an entire optimized exploration. For a tutorial on using pre-existing workflows with FireWorks, go :doc:`here. </basic>`