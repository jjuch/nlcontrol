"""
SIGNAL LIBRARY (:mod: `nlcontrol.signals')
==========================================

.. currentmodule:: nlcontrol.signals

Functions:
    * signal_constructors : 
        step : An N-channel step signal
        sinusoid: An N-channel sinusoid signal
        impulse : An N-channel impulse signal
        empty_signal: An N-channel zero signal
    * signal_tools : 
        append : Append multiple signals to one signal
        add : Add the output of two signals together

"""

from .signal_constructors import step, empty_signal, sinusoid, impulse
from .signal_tools import append, add