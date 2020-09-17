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
    * eulaC :
        EulerLagrangeController : A conversion class build upon the DynamicController class
"""

from .controller import ControllerBase, DynamicController
from .basic import PID
from .eulaC import EulerLagrangeController