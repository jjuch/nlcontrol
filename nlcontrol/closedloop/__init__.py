"""
CLOSEDLOOP LIBRARY (:mod: `nlcontrol.closedloop')
=================================================

.. currentmodule:: nlcontrol.closedloop

Classes:
    * feedback : 
        ClosedLoop : A closed-loop control scheme simulator.

Functions:
    * blocks : 
        gain_block : Multiply signals with constant.
        summation_block :  Sum two signals together.

"""

from .feedback import ClosedLoop
from .blocks import gain_block, summation_block