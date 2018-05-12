.. title:: rocketsled

.. image:: _static/rsfw.png
   :width: 600 px
   :alt: rocketsled logo
   :align: center

rocketsled is a flexible, automatic
`(open source) <https://github.com/hackingmaterials/rocketsled>`_ optimization
framework *"on rails"* for serious high throughput computation.

rocketsled is an extension of
`FireWorks <https://github.com/materialsproject/fireworks>`_ workflow software,
written in Python.


=========================
Is rocketsled for me?
=========================
Is your computational problem:

1. Expensive and/or complex?
----------------------------
    **Does it require high performance computing +  workflow tools?**

2. Run in high-throughput?
--------------------------
    **Many similar workflows running concurrently?**

3. Limited by an allocation of CPU hours?
-----------------------------------------
    **Want the most "bang for your buck"?**


If you answered yes to these three questions, *keep reading!*

If you have a complex, multi-iteration task to execute on high performance
computers, and you would like to *automatically* reduce the number of expensive
calculations needed to run your task, **rocketsled is for you.**

============================
What does rocketsled do?
============================

``rocketsled`` functions as a **black box optimizer** for a sequential optimization
loop; it requires no knowledge of a function in order to optimize it.

More importantly though, rocketsled **retains the workflow management abilties** of FireWorks (provenance, dynamic workflows, duplicate detection and correction,
error handling) across **arbitrary computing resources**.

``rocketsled`` is intended to be "plug-and-play": simply "plug-in" an expensive
objective function or workflow, constraints, and (optionally) a black box optimization algorithm,
and rocketsled + FireWorks automatically creates the optimization loop.


A visual explanation...
-----------------------

``rocketsled`` is designed for optimization tasks with sequential improvement. For example, a typical workflow without optimization might look like this:

.. image:: _static/singlewf.png
   :alt: basicwf
   :align: center

Input parameters are given to the first job (Firework). This begins the workflow, and a useful output result is given. The workflow is repeated as desired, often across many compute nodes in parallel.

.. image:: _static/miniwf.png
   :alt: basicwf
   :width: 150px
.. image:: _static/miniwf.png
   :alt: basicwf
   :width: 150px
.. image:: _static/miniwf.png
   :alt: basicwf
   :width: 150px
.. image:: _static/miniwf.png
   :alt: basicwf
   :width: 150px

Randomly selecting the next sets of input parameters to run is *inefficient*, since we will execute many workflows, including those with unfavorable results. To reduce the required number of computed workflows, we need to *intelligently* choose new input parameters with an **optimization loop.**

This is where ``rocketsled`` comes in handy. ``rocketsled`` is a sub-job (FireTask) called ``OptTask``, which can go in any Firework in the workflow, and which uses ``sklearn`` regressors to predict the best *input* parameters for the next iteration, store them in a MongoDB database, and automatically submit a new workflow to compute the next output.

.. image:: _static/singlewf_withrs.png
   :alt: basicwf
   :align: center


Features of ``rocketsled``
--------------------------

* One-line setup tools

* Persistent storage and optimization tracking

* Automatic workflow submission and management with FireWorks

* Ability to handle complex search spaces, including:
    + discrete (categorical, integer) dimensions
    + continuous dimensions
    + discontinuous spaces (subsets of entire spaces)

* 10 Built-in "out-of-the-box" sklearn-based tunable Bayesian optimizers
    + single objective
    + multi objective

* Support for nearly any custom optimizer written in Python (Bayesian and non-Bayesian)

* Facilitated feature engineering with ``get_z`` argument

* Tuneable control of training and prediction performance, across many kinds of computer resources

* Avoids submitting duplicate workflows, even when workflows run with massive parallelism

* Customization of optimization scheme (sequential, batch, etc.)

* Automatic hyperparameter optimization

* Automatic encoding for categorical optimization

* and more... (see comprehensive guide)


============
Installation
============


Requirements
------------

* Python 2 or 3
* NumPy
* SciPy
* Scikit-learn
* FireWorks
* MongoDB


Install
-------

.. code-block:: bash

    $ # Download the repository and install
    $ git clone https://github.com/hackingmaterials/rocketsled.git
    $ cd rocketsled
    $ pip install -e . -r requirements.txt


Run tests locally
-----------------

.. code-block:: bash

    $ # Make sure the mongod daemon is running as admin
    $ mongod
    $ python setup.py test

Tip: To run tests using a remote launchpad, edit ``/rocketsled/tests/tests_launchpad.yaml``

=========
Tutorials
=========

*Tutorials 1 and 2 require some knowledge of Fireworks. If you aren't comfortable with Fireworks, please work through the tutorials* `here <https://hackingmaterials.lbl.gov/fireworks/>`_.


:doc:`Tutorial 0 - Quickstart </quickstart>`
-----------------------

In the quickstart, we show how to use rocketsled's ``auto_setup`` to get
up and running quickly, starting only with an objective function written in Python.

**Time to complete**: 5 min

:doc:`Tutorial 1 - Basic usage </basic>`
------------------------

In this tutorial we demonstrate how to get a basic optimization loop up and
running with a FireWorks workflow.

**Time to complete**: 5 min


:doc:`Tutorial 2 - Advanced usage </advanced>`
---------------------------
In this tutorial we explore the more advanced capabilities of ``OptTask``.

**Time to complete**: 20 min


:doc:`A Comprehensive Guide to rocketsled </guide>`
---------------------------------------

Find a comprehensive guide to using rocketsled at the link below. The guide
exhaustively documents the possible arguments to ``OptTask`` and provides at least
one example of each. If working through the tutorials did not answer your
question, you'll most likely find your answer here.

Documentation
-------------

Find the auto-generated documentation :doc:`here </modules>`. Beware! Only for the brave.


===========
Use Cases
===========

Rocketsled is applicable to many types of scientific computing problems.




