from systems.system import SystemBase
from simupy.systems import SystemFromCallable
from sympy.tensor.array import Array
import numpy as np

def step(*args, **kwargs):
    """
    Creates a BaseSystem class object with a step signal. The signal can consist of multiple channels. This makes it possible to connect the step system to a system with multiple inputs. Without any arguments a one-channel unity step at time instance 0 is returned. By adding a dimension (int) N an N-channel unity step at time instance 0 is returned. The keyword arguments allow the creation of more customization: step_times defines the time the step starts (in seconds), begin_values defines the start values, and end_values defines the stop values. These three keyword arguments need to have the same dimension and are defined as lists or as an int. Each keyword is optional and left blank defaults to its unity step value.

    Parameters:
    -----------
        args : int, optional
            the number of channels, default: 1
        kwargs : list or int, optional
            step_times : time in seconds that the step starts, default: 0.
            begin_values : begin values of the step, default: 0.
            end_values : end values of the step, default: 1.

    Returns:
    --------
        A SystemBase object with parameters:
            states : NoneType
                None
            inputs : NoneType
                None
            sys : SystemFromCallable object 
                with a function 'callable' returning a numpy array in function of time t, 0 inputs, and dim outputs.

    Example:
    -------
        * 1-channel unity step at second 0:
        >>> step_signal = step()

        * 2-channel unity step, both start at second 0:
        >>> step_signal = step(2) 

        * 2-channel unity steps where channel 1  starts at second 2 and channel 2 starts at 5 seconds:
        >>> step_signal = step(step_times=[2, 5]) 
        
        * 2-channel step with channel 1 steps from 1 to 3 at time 0, and channel 2 steps from 2 to 2.5 at time 0:
        >>> step_signal = step(begin_values=[1, 2], end_values=[3, 2.5])
    """

    def _check_inputs(dim, kwargs):
        test_kwargs = [1 == dim if isinstance(value, int)  else len(value) == dim for key, value in kwargs.items()]
        if False in test_kwargs:
            error_text = '[signals.step] the inputs have different dimensions.'
            raise ValueError(error_text)


    dim = 1
    step_times = dim * [0]
    begin_values = dim * [0]
    end_values = dim * [1]
    if len(args) == 1:
        dim = args[0]
        if len(kwargs) != 0:
            _check_inputs(dim, kwargs)
            if ("step_times" in kwargs):
                step_times = kwargs.step_times
            if ("begin_values" in kwargs):
                begin_values = kwargs.begin_values
            if ("end_values" in kwargs):
                end_values = kwargs.end_values
    elif len(args) == 0 and len(kwargs) != 0:
        a_key = list(kwargs.items())[0][0]
        a_list = kwargs[a_key]
        if (not isinstance(a_list, int)):
            dim = len(a_list)
            step_times = dim * [0]
            begin_values = dim * [0]
            end_values = dim * [1]
            _check_inputs(dim, kwargs)
            if ("step_times" in kwargs):
                step_times = kwargs['step_times']
            if ("begin_values" in kwargs):
                begin_values = kwargs['begin_values']
            if ("end_values" in kwargs):
                end_values = kwargs['end_values']
        else:
            _check_inputs(dim, kwargs)
            if ("step_times" in kwargs):
                step_times = [kwargs['step_times']]
            if ("begin_values" in kwargs):
                begin_values = [kwargs['begin_values']]
            if ("end_values" in kwargs):
                end_values = [kwargs['end_values']]
    elif len(args) > 1:
        error_text = '[signals.step] Too many arguments without a keyword.'
        raise AssertionError(error_text)


    def callable(t, *args):
        mask = [t - el >= 0 for el in step_times]
        values = np.array([end_values[index] if mask_val else begin_values[index] for index, mask_val in enumerate(mask)])
        return values

    return SystemBase(states=None, inputs=None, sys=SystemFromCallable(callable, 0, dim))
        