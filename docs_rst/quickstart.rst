=======================================
Welcome to the `rocketsled` quickstart!
=======================================

*Note: This quickstart assumes a limited knowledge of FireWorks. If you already have a workflow built, see the examples or the more advanced tutorials.*

The easiest way to get started is to use rocketsled's `auto_setup`. `auto_setup` wraps any Python function in a FireWork, adds a Firework containing an OptTask optimization, and creates a workflow ready for launch.

Let's get an optimization running on your local machine. First, make sure a `mongod` instance is running.

.. code-block:: bash

    $ mongod


Define objective function
-------------------------

Great! Now lets define a trivial objective function f(x) for this demo. Your actual objective function will be **much** more complex than this.

.. code-block:: python
    # The objective function must accept a vector and return a scalar.
    def f(x):
        return x[0] * x[1] / x[2]


