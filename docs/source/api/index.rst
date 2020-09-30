API
====

Here, you can find all information on the different classes, definitions, etc. of the `nlcontrol` module. There are three main classes: SystemBase, ControllerBase, and ClosedLoop. Next to these base classes, there are more advanced system and controller classes. This list is far from completed. If you created a new controller or system based on the base classes, do not hesitate to contribute it to this toolbox to help humankind.

The Idea
--------

The advantage of using this SystemBase and ControllerBase classes is that it can easily be implemented in a closed loop configuration with another SystemBase and/or controllerBase object.

This toolbox is strongly based on the `SimuPy <https://simupy.readthedocs.io/>`__ module. The contribution of this module is to create a more accessible nonlinear control toolbox, which can be used by proficient Python programmers as well as for users who do not want to focus on programming at all.

The Docs
--------

    .. toctree::
        :maxdepth: 2

        systems
        controllers
        closed_loop