import nlcontrol.systems as nlSystems

from simupy.systems import SystemFromCallable

from sympy.tensor.array import Array
from sympy import Symbol

import numpy as np

def step(dim=None, step_times=None, begin_values=None, end_values=None):
    """
    Creates a BaseSystem class object with a step signal. The signal can consist of multiple channels. This makes it possible to connect the step system to a system with multiple inputs. Without any arguments a one-channel unity step at time instance 0 is returned. By adding a dimension (int) N an N-channel unity step at time instance 0 is returned. The keyword arguments allow the creation of more customization: step_times defines the time the step starts (in seconds), begin_values defines the start values, and end_values defines the stop values. These three keyword arguments need to have the same dimension and are defined as lists or as an int. Each keyword is optional and left blank defaults to its unity step value.

    Parameters:
    -----------
        dim : int, optional
            the number of channels, default: 1
        step_times : list or int, optional
            time in seconds when the step starts, default: 0.
        begin_values : list or int, optional
            begin values of the step, default: 0.
        end_values : list or int, optional
            end values of the step, default: 1.

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
        * 1-channel unity step at second 0:
        >>> step_signal = step()

        * 2-channel unity step, both start at second 0:
        >>> step_signal = step(2) 

        * 2-channel unity steps where channel 1  starts at second 2 and channel 2 starts at 5 seconds:
        >>> step_signal = step(step_times=[2, 5]) 
        
        * 2-channel step with channel 1 steps from 1 to 3 at time 0, and channel 2 steps from 2 to 2.5 at time 0:
        >>> step_signal = step(begin_values=[1, 2], end_values=[3, 2.5])
    """
    def _check_inputs(dim_arg, **kwargs):
        dims = [0 if value is None else 1 if np.isscalar(value) else len(value) for key, value in kwargs.items()]
        if dim_arg is not None:
            dims.append(dim_arg)
        else:
            dims.append(0)
        dims_filter = list(filter(lambda x: x != 0, dims))
        if not all(dims_filter[0] == item for item in dims_filter):
            error_text = '[signals.step] the inputs have different dimensions.'
            raise ValueError(error_text)

        #defaults
        dim = 1 if len(dims_filter) == 0 else dims_filter[0]
        step_times = dim * [0]
        begin_values = dim * [0]
        end_values = dim * [1]
        if kwargs['step_times'] is not None:
            step_times = [kwargs['step_times']] if np.isscalar(kwargs['step_times']) else kwargs['step_times']
            dim = len(step_times)
        if kwargs['begin_values'] is not None:
            begin_values = [kwargs['begin_values']] if np.isscalar(kwargs['begin_values']) else kwargs['begin_values']
            dim = len(begin_values)
        if kwargs['end_values'] is not None:
            end_values = [kwargs['end_values']] if np.isscalar(kwargs['end_values']) else kwargs['end_values']
            dim = len(end_values)

        return dim, step_times, begin_values, end_values
    
    dim, step_times, begin_values, end_values = \
        _check_inputs(dim, step_times=step_times, \
            begin_values=begin_values, end_values=end_values)
    
    def callable(t, *args):
        mask = [t - el >= 0 for el in step_times]
        values = np.array([end_values[index] if mask_val else begin_values[index] for index, mask_val in enumerate(mask)])
        return values

    system = SystemFromCallable(callable, 0, dim)
    return nlSystems.SystemBase(states=None, inputs=None, sys=system)


def sinusoid(dim=None, amplitude=None, frequency=None, phase_shift=None, y_shift=None):
    """
    Creates a BaseSystem class object with a sinusoid signal. The signal can consist of multiple channels. This makes it possible to connect the sinusoid system to a system with multiple inputs. Without any arguments a one-channel sinusoid with amplitude 1, frequency 1 Hz,  phase shift 0 rad and 0 y-shift is returned. By adding a dimension (int) N an N-channel sinusoid with amplitude 1, frequency 1 Hz, phase shift 0 rad, and y-shift 0 is returned. The keyword arguments allow the creation of more customization: 'amplitude' defines the amplitude of the sines, 'frequency' defines the frequency in Hz, 'phase_shift' defines the phase shift in radians, and 'y_shift' defines the shift of the sine compared to the y-axis. These four keyword arguments need to have the same dimension and are defined as lists or as an int. Each keyword is optional and left blank defaults to amplitude 1, frequency 1 Hz, phase shift 0 rad, and y-shift 0. For the sinusoid, numpy's sin function is used.

    Parameters:
    -----------
        dim : int, optional
            the number of channels, default: 1
        amplitude : list or int, optional
            amplitude of sinusoid, default: 1.
        frequency : list or int, optional
            frequency in Hz of the sinusoid, default: 1.
        phase_shift : list or int, optional
            phase shift in radians of sinusoid, default: 0.
        y_shift : list or int, optional
            shift of the sinusoid on y-axis, default: 0.

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
        * 1-channel sinusoid sin(2.pi.t):
        >>> sin_signal = sinusoid()

        * 2-channel sinusoids sin(2.pi.t):
        >>> sin_signal = sinusoid(2) 

        * 2-channel sinusoids [2.sin(2.pi.t), 5.sin(2.pi.t)]:
        >>> sinusoid_signal = sinusoid(amplitude=[2, 5]) 
        
        * 2-channel sinusoids [sin(2.pi.t + 3) + 2, sin(2.pi.2.t + 2.5) - 1]:
        >>> sinusoid_signal = sinusoid(frequency=[1, 2], phase_shift=[3, 2.5], y_shift=[2, -1])
    """
    def _check_inputs(dim_arg, **kwargs):
        dims = [0 if value is None else 1 if np.isscalar(value) else len(value) for key, value in kwargs.items()]
        if dim_arg is not None:
            dims.append(dim_arg)
        else:
            dims.append(0)
        dims_filter = list(filter(lambda x: x != 0, dims))
        if not all(dims_filter[0] == item for item in dims_filter):
            error_text = '[signals.sinusoid] the inputs have different dimensions.'
            raise ValueError(error_text)

        #defaults
        dim = dim = 1 if len(dims_filter) == 0 else dims_filter[0]
        amplitude = dim * [1]
        frequency = dim * [1]
        phase_shift = dim * [0]
        y_shift = dim * [0]
        if kwargs['amplitude'] is not None:
            amplitude = [kwargs['amplitude']] if np.isscalar(kwargs['amplitude']) else kwargs['amplitude']
            dim = len(amplitude)
        if kwargs['frequency'] is not None:
            frequency = [kwargs['frequency']] if np.isscalar(kwargs['frequency']) else kwargs['frequency']
            dim = len(frequency)
        if kwargs['phase_shift'] is not None:
            phase_shift = [kwargs['phase_shift']] if np.isscalar(kwargs['phase_shift']) else kwargs['phase_shift']
            dim = len(phase_shift)
        if kwargs['y_shift'] is not None:
            y_shift = [kwargs['y_shift']] if np.isscalar(kwargs['y_shift']) else kwargs['y_shift']
            dim = len(y_shift)

        return dim, amplitude, frequency, phase_shift, y_shift
    
    dim, amplitude, frequency, phase_shift, y_shift = \
        _check_inputs(dim, amplitude=amplitude, \
            frequency=frequency, phase_shift=phase_shift, y_shift=y_shift)

    def callable(t, *args):
        return np.array([A * np.sin(2*np.pi*f*t + phi) + y for A, f, phi, y in zip(amplitude, frequency, phase_shift, y_shift)])
    
    system = SystemFromCallable(callable, 0, dim)
    return nlSystems.SystemBase(states=None, inputs=None, sys=system)


def impulse(dim=None, amplitude=None, impulse_time=None, eps=10**(-2)):
    """
    Creates a BaseSystem class object with a semi-impulse signal:
        u(t) = A    if abs(t - ts) < eps
             = 0    elsewhere 
    with A amplitude, ts the impulse time, and eps a small number.
    The signal can consist of multiple channels. This makes it possible to connect the impulse system to a system with multiple inputs. Without any arguments a one-channel impulse with amplitude 1 triggered at 0s is returned. By adding a dimension (int) N an N-channel impulse with amplitude 1, triggered at 0s is returned. The keyword arguments allow the creation of more customization: 'amplitude' defines the amplitude of the impulses, 'impulse_time' defines when the impulse is triggered, and eps defines how narrow the semi-impulse should be. The keyword arguments 'amplitude' and 'impulse_time' need to have the same dimension and are defined as lists or as an int. Each keyword is optional and left blank defaults to amplitude 1, and impulse time 0s.

    Parameters:
    -----------
        dim : int, optional
            the number of channels, default: 1
        amplitude : list or int, optional
            amplitude of impulses, default: 1.
        impulse_time : list or int, optional
            the times the impulse is triggered in seconds, default: 0.
        eps : float, optional
            the narrowness of the semi-impulse, default: 10^(-2).

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
        * 1-channel default impulse:
        >>> impulse_signal = impulse()

        * 2-channel default impulse:
        >>> impulse_signal = impulse(2) 

        * 2-channel impulse with amplitude 2 and 5 repectively:
        >>> impulse_signal = impulse(amplitude=[2, 5]) 
        
        * plot a 2-channel impulse with default amplitudes, triggered at 2s and 3.5s respectively, and a band of 3 * 10^(-4):
        >>> impulse_signal = impulse(impulse_time=[2, 3.5], eps=1.5*10**(-4))
        >>> impulse_signal.simulation(5, number_of_samples=10000, plot=True)
    """
    def _check_inputs(dim_arg, **kwargs):
        dims = [0 if value is None else 1 if np.isscalar(value) else len(value) for key, value in kwargs.items()]
        if dim_arg is not None:
            dims.append(dim_arg)
        else:
            dims.append(0)
        dims_filter = list(filter(lambda x: x != 0, dims))
        if not all(dims_filter[0] == item for item in dims_filter):
            error_text = '[signals.impulse] the inputs have different dimensions.'
            raise ValueError(error_text)

        #defaults
        dim = dim = 1 if len(dims_filter) == 0 else dims_filter[0]
        amplitude = dim * [1]
        impulse_time = dim * [0]
        if kwargs['amplitude'] is not None:
            amplitude = [kwargs['amplitude']] if np.isscalar(kwargs['amplitude']) else kwargs['amplitude']
            dim = len(amplitude)
        if kwargs['impulse_time'] is not None:
            impulse_time = [kwargs['impulse_time']] if np.isscalar(kwargs['impulse_time']) else kwargs['impulse_time']
            dim = len(impulse_time)

        return dim, amplitude, impulse_time
    
    dim, amplitude, impulse_time = \
        _check_inputs(dim, amplitude=amplitude, \
            impulse_time=impulse_time)
    
    def callable(t, *args): 
        return np.array([A if abs(t - t_shift) < eps else 0  \
            for A, t_shift in zip(amplitude, impulse_time)])
    
    system = SystemFromCallable(callable, 0, dim)
    return nlSystems.SystemBase(states=None, inputs=None, sys=system)


def empty_signal(dim):
    """
    Creates a BaseSystem class object with a zero signal. The signal can consist of multiple channels. This makes it possible to connect a zero input to a system if needed.

    Parameters:
    -----------
        dim: int
            The number of channels.

    Returns:
    --------
        A SystemBase object with parameters:
            states : NoneType
                None
            inputs : NoneType
                None
            sys : SystemFromCallable object 
                with a function 'callable' returning a numpy array with zeros, 0 inputs, and dim outputs.
    """
    if (dim == 0):
        dim = 1

    def callable(t, *args):
        return np.array(dim * [0])

    system = SystemFromCallable(callable, 0, dim)
    return nlSystems.SystemBase(states=None, inputs=None, sys=system)


      