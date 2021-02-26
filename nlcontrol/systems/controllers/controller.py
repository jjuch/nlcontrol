from nlcontrol.systems import SystemBase 
import nlcontrol.systems.controllers.utils as nlctr_ctr_utils

from sympy.matrices import Matrix
from sympy.tensor.array import Array
from sympy import diff, Symbol, integrate
from sympy.core.function import Derivative

from simupy.systems.symbolic import DynamicalSystem

import numpy as np

__all__ = ["ControllerBase"]

class ControllerBase(SystemBase):
    """
    ControllerBase(states, inputs, sys=None, name="controller")

    Returns a base structure for a controller with outputs, optional inputs, and optional states. The controller is an instance of a SystemBase, which is defined by it state equations (optional):
    
    .. math::
        \\frac{dx(t)}{dt} = h(x(t), u(t), t)

    with x(t) the state vector, u(t) the input vector and t the time in seconds. Next, the output is given by the output equation:

    .. math::
        y(t) = g(x(t), u(t), t)

    Parameters
    -----------
    states : string or array-like
        if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols.
    inputs : string or array-like
        if `inputs` is a string, it is a comma-separated listing of the input names. If `inputs` is array-like it contains the inputs as sympy's dynamic symbols.
    system : simupy's DynamicalSystem object (simupy.systems.symbolic), optional
        the object containing output and state equations, default: None.
    name : string
        give the system a custom name which will be shown in the block scheme, default: 'controller'.

    Examples
    ---------
    * Statefull controller with one state, one input, and one output:
        >>> from simupy.systems.symbolic import MemorylessSystem, DynamicalSystem
        >>> from sympy.tensor.array import Array
        >>> st = 'z'
        >>> inp = 'w'
        >>> contr = ControllerBase(states=st, inputs=inp)
        >>> z, zdot, w = contr.create_variables()
        >>> contr.system = DynamicalSystem(state_equation=Array([-z + w]), state=z, output_equation=z, input_=w)

    * Statefull controller with two states, one input, and two outputs:
        >>> st = 'z1, z2'
        >>> inp = 'w'
        >>> contr = ControllerBase(states=st, inputs=inp)
        >>> z1, z2, z1dot, z2dot, w = contr.create_variables()
        >>> contr.system = DynamicalSystem(state_equation=Array([-z1 + z2**2 + w, -z2 + 0.5 * z1]), state=Array([z1, z2]), output_equation=Array([z1 * z2, z2]), input_=w)

    * Stateless controller with one input:
        >>> st = None
        >>> inp = 'w'
        >>> contr = ControllerBase(states=st, inputs=inp)
        >>> w = contr.create_variables()
        >>> contr.system = MemorylessSystem(input_=Array([w]), output_equation= Array([5 * w]))

    * Create a copy a ControllerBase object 'contr' and linearize around the working point of state [0, 0] and working point of input 0 and simulate:
        >>> new_contr = ControllerBase(states=contr.states, inputs=contr.inputs, sys=contr.system)
        >>> new_contr_lin = new_contr.linearize([0, 0], 0)
        >>> new_contr_lin.simulation(10)
    """

    def __init__(self, *args, **kwargs):
        if 'inputs' in kwargs.keys():
            inputs = kwargs['inputs']
        else:
            error_text = "[nlcontrol.systems.ControllerBase] An 'inputs=' keyword is necessary."
            raise AssertionError(error_text)
        if 'states' in kwargs.keys():
            states = kwargs['states']
        else:
            states = None
        if 'system' in kwargs.keys():
            sys = kwargs['system']
        else:
            sys = None
        if 'name' in kwargs.keys():
            name = kwargs['name']
        else:
            name = "controller"
        super().__init__(states, inputs, system=sys, name=name, block_type="controller")
        self.dinputs, self.iinputs = self.__create_inputs__()

    
    def __create_inputs__(self):
        """
        Create lists of differentiated and integrated symbols of the input vector.

        Returns
        --------
        variables : tuple 
            inputs_diff : MDimArray
                a list of differentiated input symbols.
            inputs_int : MDimArray
                a list of integrated input symbols.
        """
        inputs_diff = [diff(input_el, Symbol('t')) for input_el in self.inputs]
        inputs_int = [integrate(input_el, Symbol('t')) for input_el in self.inputs]
        return inputs_diff, inputs_int

    
    def series(self, contr_append):
        """
        A controller is generated which is the result of a serial connection of two controllers. The outputs of this object are connected to the inputs of the appended system and a new controller is achieved which has the inputs of the current system and the outputs of the appended system. Notice that the dimensions of the output of the current system should be equal to the dimension of the input of the appended system.

        Parameters
        -----------
        contr_append : ControllerBase object
            the controller that is placed in a serial configuration. 'contr_append' follows the current system.

        Returns
        --------
        A ControllerBase object with the serial system's equations.

        Examples
        ---------
        * Place 'contr1' behind 'contr2' in a serial configuration and show the inputs, states, state equations and output equations:
            >>> series_sys = contr1.series(contr2)
            >>> print('inputs: ', series_sys.system.input_)
            >>> print('States: ', series_sys.system.state)
            >>> print('State eqs: ', series_sys.system.state_equation)
            >>> print('Output eqs: ', series_sys.system.output_equation)
        """
        series_system = super().series(contr_append)
        return nlctr_ctr_utils.toControllerBase(series_system)
        # return ControllerBase(inputs=series_system.inputs, states=series_system.states, system=series_system.system)

    
    def parallel(self, contr_append):
        """
        A controller is generated which is the result of a parallel connection of two controllers. The inputs of this object are connected to the system that is placed in parallel and a new system is achieved with the output the sum of the outputs of both systems in parallel. Notice that the dimensions of the inputs and the outputs of both systems should be equal.

        Parameters
        -----------
        contr_append : ControllerBase object
            the controller that is added in parallel.

        Returns
        --------
        A ControllerBase object with the parallel system's equations.

        Examples
        ---------
        * Place 'contr2' in parallel with 'contr1' and show the inputs, states, state equations and output equations:
        >>> parallel_sys = contr1.parallel(contr2)
        >>> print('inputs: ', parallel_sys.system.input_)
        >>> print('States: ', parallel_sys.system.state)
        >>> print('State eqs: ', parallel_sys.system.state_equation)
        >>> print('Output eqs: ', parallel_sys.system.output_equation)
        """
        parallel_system = super().parallel(contr_append)
        return nlctr_ctr_utils.toControllerBase(parallel_system)
        # return ControllerBase(inputs=parallel_system.inputs, states=parallel_system.states, system=parallel_system.system)