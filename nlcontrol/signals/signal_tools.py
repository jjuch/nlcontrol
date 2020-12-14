import nlcontrol.systems as nlSystems
import numpy as np
from simupy.systems import DynamicalSystem, SystemFromCallable

__all__ = ["append", "add"]

def append(*signals):
    """
    Append a N_i-channel signals to a sum(N_i, i)-channel signal. Add as many signals as needed. The order of appearance determines the index of the output.

    Parameters:
    -----------
        signals: nlcontrol.signals object
            the signals that need to be appended to a new signal.
    
    Returns: 
    --------
        A SystemBase object with parameters:
            states : NoneType
                None
            inputs : NoneType
                None
            sys : SystemFromCallable object 
                with a function 'callable' returning a numpy array in function of time t, 0 inputs, and dim outputs.

    Examples:
    ---------
        * Append steps and sinusoids into one system:
        >>> step1_sig = step(step_times=[2.5, 3.5], begin_values=[1.8, 0.5], end_values=[-1.2, 2.1])
        >>> step2_sig = step(step_times=0.2, begin_values=-2.3, end_values=0.8)
        >>> sin_sig = sinusoid(amplitude=2.3, frequency=1.5, phase_shift=-0.3)
        >>> appended_signal = append(step1_sig, sin_sig, step2_sig)
        >>> appended_signal.simulation(5, plot=True)
    """
    if len(signals) == 1:
        error_text = '[signals.append] You need at least two signals to append.'
        raise AssertionError(error_text)
    
    dim = 0
    for i in range(len(signals)):
        signal = signals[i].system
        dim += signal.dim_output
        if not isinstance(signal, DynamicalSystem):
            error_text = '[signals.append] Only append signal objects.'
            raise AssertionError(error_text)

    def callable(t, *args):
        values = np.array([])
        for signal in signals:
            values = np.append(values, signal.system.output_equation_function(t))
        return values
    
    system = SystemFromCallable(callable, 0, dim)
    return nlSystems.SystemBase(states=None, inputs=None, sys=system)

def add(signal1, signal2):
    """
    Add the channels of signal1 to the channels of signal2. Be aware that the dimensions of the outputs of both signals should be the same.
        y1 = [y1_1, y1_2, ..., y1_n]
        y2 = [y2_1, y2_2, ..., y2_n]
        y = y1 + y2 = [y1_1 + y2_1, y1_2 + y2_2, ..., y1_n + y2_n]

    Parameters:
    -----------
        signal1: nlcontrol.signals object
            the first signal that will be added.
        signal2: nlcontrol.signals object
            the second signal that will be added.
    
    Returns: 
    --------
        A SystemBase object with parameters:
            states : NoneType
                None
            inputs : NoneType
                None
            sys : SystemFromCallable object 
                with a function 'callable' returning a numpy array in function of time t, 0 inputs, and dim outputs.

    Examples:
    ---------
        * Create two sinusoids, the average value has to change after 1.5 and 3.5s respectively:
        >>> sin_sig = sinusoid(2)
        >>> step_sig = step(step_times=[1.5, 3.5])
        >>> added_signal = add(sin_sig, step_sig)
        >>> added_signal.simulation(5, plot=True)
    """
    if not isinstance(signal1.system, DynamicalSystem):
        error_text = '[signals.add] signal1 should be a nlcontrol.signals object.'
        raise AssertionError(error_text)
    if not isinstance(signal2.system, DynamicalSystem):
        error_text = '[signals.add] signal2 should be a nlcontrol.signals object.'
        raise AssertionError(error_text)
    if signal1.system.dim_output != signal2.system.dim_output:
        error_text = '[signals.add] The output dimension of signal1 and signal2 should be equal.'
        raise AssertionError(error_text)
    else:
        dim = signal1.system.dim_output

    def callable(t, *args):
        return np.array([s1 + s2 for s1, s2 \
            in zip(signal1.system.output_equation_function(t), signal2.system.output_equation_function(t))])

    system = SystemFromCallable(callable, 0, dim)
    return nlSystems.SystemBase(states=None, inputs=None, sys=system)