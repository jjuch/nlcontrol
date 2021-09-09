"""
CONTROLLER LIBRARY (:mod: `nlcontrol.systems.controller')
=========================================================

..currentmodule:: nlcontrol.systems.controller

Classes:
    * controller :
        ControllerBase : The base definition of any type of controller, based on SystemBase object.
    * basic : 
        PID : A nonlinear PID controller.
    * circle_criterium : controllers based on the circle criterium
        DynamicController : A general dynamic controller definition.
        EL_circ : A Euler-Lagrange conversion class build upon the DynamicController class

Functions:
    * utils : 
        toControllerBase : Transform a SystemBase object to a ControllerBase object.
"""

from . import controller
from .controller import ControllerBase
from . import basic
from .basic import PID
from . import circle_criterium
from .circle_criterium import DynamicController, EL_circ
from . import utils
from .utils import toControllerBase