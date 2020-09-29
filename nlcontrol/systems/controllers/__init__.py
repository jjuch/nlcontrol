"""
CONTROLLER LIBRARY (:mod: `nlcontrol.systems.controller')
=========================================================

..currentmodule:: nlcontrol.systems.controller

Classes:
    * controller :
        ControllerBase : The base definition of any type of controller, based on SystemBase object.
    * basic : 
        PID : A nonlinear PID controller.
    * eulaC :
        DynamicController : A general dynamic controller definition.
        EulerLagrangeController : A conversion class build upon the DynamicController class
"""

from .controller import ControllerBase
from .basic import PID
from .eulaC import EulerLagrangeController, DynamicController