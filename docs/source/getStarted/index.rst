.. _getStarted:

============
Get Started
============

.. only:: html

    .. contents::
       :depth: 3
       :backlinks: none
       

Installation
------------

The installation procedure requires Python 3. Some additional packages are required and are installed upon installation of the `nlcontrol`. Currently, only pip is available.

pip
^^^^
If you use `pip` you can install the package as follows::

    pip install nlcontrol

.. warning:: the dependency module `python-control <https://python-control.readthedocs.io/>`__ has an optional dependency `slycot`, which should be installed separately. More info can be found `here <https://python-control.readthedocs.io/en/0.8.3/intro.html#installation>`__.


Current Release
^^^^^^^^^^^^^^^^

* **2020-10-01** `nlcontrol-1.0.1.tar.gz`_

.. _`nlcontrol-1.0.1.tar.gz`: https://github.com/jjuch/nlcontrol/releases/download/v1.0.1/nlcontrol-1.0.1.tar.gz

Past Releases
^^^^^^^^^^^^^^

*None*


.. _source:

Development Source
^^^^^^^^^^^^^^^^^^^

The main repository for `nlcontrol` is located on github at
https://github.com/jjuch/nlcontrol.

You can obtain a copy of the active source code by issuing the following
command

::

    git clone https://github.com/jjuch/nlcontrol.git




Usage
------
Import the module in your Python code by using the following statement::

    import nlcontrol

To import specific parts of the `nlcontrol` module use the following statement::
    
    from nlcontrol import < *what-you-want-to-import* >