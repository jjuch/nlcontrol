"""
SYSTEMS LIBRARY (:mod: `nlcontrol.systems')
===========================================

.. currentmodule:: nlcontrol.systems

Classes:
    * system :
        SystemBase : The base definitions of each system object.
    * eula : 
        EulerLagrange : An Euler-Lagrange notation of a system.
    * controller : 
        PID: A nonlinear PID controller formulation.

Functions:
    * utils:
        write_simulation_result_to_csv : write a SimulationResult or list of result vectors to a csv
        read_simulation_result_from_csv : read a csv created by write_simulation_result_to_csv

"""

from .system import SystemBase
from .eula import EulerLagrange
from .controller import PID
from .utils import write_simulation_result_to_csv, read_simulation_result_from_csv