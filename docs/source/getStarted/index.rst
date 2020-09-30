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

* **2020-09-01** `nlcontrol-1.0.3.tar.gz`_

.. _`gdal-3.1.3.tar.gz`: https://github.com/jjuch/nlcontrol/releases/download/v1.0.3/nlcontrol-1.0.3.tar.gz

Past Releases
^^^^^^^^^^^^^^

*None*




Usage
------
Import the module in your Python code by using the following statement::

    import nlcontrol

To import specific parts of the `nlcontrol` module use the following statement::
    
    from nlcontrol import < *what you want to import* >