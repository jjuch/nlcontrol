"""
SYSTEMS LIBRARY (:mod: `nlcontrol.systems')
===========================================

.. currentmodule:: nlcontrol.systems

Classes:
    BaseSystem : The base definitions of each system object.
    EulerLagrange : An Euler-Lagrange notation of a system.
    UMS : An underactuated mechanical system formulation based on EulerLagrange.
    Controller: An Euler-lagrange based controller.
"""

from .system import BaseSystem
from .eula import EulerLagrange
from .ums import UMS
from .controller import Controller