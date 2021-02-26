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

Functions:
    * utils : 
        toControllerBase : Transform a SystemBase object to a ControllerBase object.
"""

from . import controller
from .controller import ControllerBase
from . import basic
from .basic import PID
from . import eulaC
from .eulaC import EulerLagrangeController, DynamicController
from . import utils
from .utils import toControllerBase