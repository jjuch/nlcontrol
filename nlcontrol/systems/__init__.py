"""
SYSTEMS LIBRARY (:mod: `nlcontrol.systems')
===========================================

.. currentmodule:: nlcontrol.systems

Classes:
    * system :
        SystemBase : The base definitions of each system object.
    * eula : 
        EulerLagrange : An Euler-Lagrange notation of a system.
    * controllers : 
        ControllerBase : The base definition of any type of controller, based on SystemBase object.
        DynamicController : A general dynamic controller definition.
        PID : A nonlinear PID controller.
        EulerLagrangeController : A conversion class build upon the DynamicController class

Functions:
    * utils:
        write_simulation_result_to_csv : write a SimulationResult or list of result vectors to a csv
        read_simulation_result_from_csv : read a csv created by write_simulation_result_to_csv
    * controllers / utils : 
        toControllerBase : Conversion from SystemBase to ControllerBase object.

"""

from .system import SystemBase, TransferFunction
from .eula import EulerLagrange
from . import controllers
from .controllers import *
from .utils import write_simulation_result_to_csv, read_simulation_result_from_csv