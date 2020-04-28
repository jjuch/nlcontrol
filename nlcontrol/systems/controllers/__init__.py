"""
CONTROLLER LIBRARY (:mod: `nlcontrol.systems.controller')
=========================================================

..currentmodule:: nlcontrol.systems.controller

Classes:
    * controller :
        ControllerBase : The base definition of any type of controller, based on SystemBase object.
        DynamicController : A general dynamic controller definition.
    * basic : 
        PID : A nonlinear PID controller.
"""

from .controller import ControllerBase, DynamicController
from .basic import PID