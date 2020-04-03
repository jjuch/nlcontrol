import nlcontrol.signals as sgnls

from copy import deepcopy, copy

from sympy.physics.mechanics import dynamicsymbols
from sympy.matrices import Matrix
from sympy.tensor.array.ndim_array import NDimArray
from sympy.physics.mechanics import msubs
from sympy import Symbol, diff
from sympy.tensor.array import Array

from simupy.block_diagram import BlockDiagram, SimulationResult
from simupy.systems.symbolic import MemorylessSystem, DynamicalSystem, lambdify_with_vector_args
from simupy.systems import DynamicalSystem as DynamicalSystem2

import numpy as np
import matplotlib.pyplot as plt

DEFAULT_INTEGRATOR_OPTIONS = {
    'name': 'dopri5',
    'rtol': 1e-6,
    'atol': 1e-12,
    'nsteps': 500,
    'max_step': 0.0
}

class SystemBase():
    """
    SystemBase(states, inputs, sys=None)

    Returns a base structure for a system with outputs, optional inputs, and optional states. The system is defines by it state equations (optional):
        diff(x(t), t) = h(x(t), u(t), t)
    with x(t) the state vector, u(t) the input vector and t the time in seconds. Next, the output is given by the output equation:
        y(t) = g(x(t), u(t), t)
    A SystemBase object contains several basic functions to manipulate and simulate the system.

    Parameters:
    -----------
        states : string or array-like
            if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols.
        inputs : string or array-like
            if `inputs` is a string, it is a comma-separated listing of the input names. If `inputs` is array-like it contains the inputs as sympy's dynamic symbols.
        sys : simupy's DynamicalSystem object (simupy.systems.symbolic), optional
            the object containing output and state equations, default: None.

    Examples:
    ---------
        * Statefull system with one state, one input, and one output:
        >>> from simupy.systems.symbolic import MemorylessSystem, DynamicalSystem
        >>> from sympy.tensor.array import Array
        >>> states = 'x'
        >>> inputs = 'u'
        >>> sys = SystemBase(states, inputs)
        >>> x, xdot, u = sys.createVariables()
        >>> sys.system = DynamicalSystem(state_equation=Array([-x1 + u1]), state=x1, output_equation=x1, input_=u1)

        * Statefull system with two states, one input, and two outputs:
        >>> states = 'x1, x2'
        >>> inputs = 'u'
        >>> sys = SystemBase(states, inputs)
        >>> x1, x2, x1dot, x2dot, u = sys.createVariables()
        >>> sys.system = DynamicalSystem(state_equation=Array([-x1 + x2**2 + u, -x2 + 0.5 * x1]), state=Array([x1, x2]), output_equation=Array([x1 * x2, x2]), input_=u)

        * Stateless system with one input:
        >>> states = None
        >>> inputs = 'w'
        >>> sys = SystemBase(states, inputs)
        >>> w = sys.createVariables()
        >>> sys.system = MemorylessSystem(input_=Array([w]), output_equation= Array([5 * w]))

        * Create a copy a SystemBase object `sys':
        >>> new_sys = SystemBase(sys.states, sys.inputs, sys.system)

    """
    def __init__(self, states, inputs, sys=None):
        self.states = self.__process_init_input(states)
        self.dstates = self.__process_init_input(states, 1)
        self.inputs = self.__process_init_input(inputs)
        self.sys = sys


    def __copy__(self):
        """
        Create a deap copy of the SystemBase object.
        """
        return deepcopy(self)

    @property
    def system(self):
        return self.sys

    @system.setter
    def system(self, system):
        self.sys = system  


    def __process_init_input(self, arg:str, level:int=0) -> Matrix:
        '''Return the correct format of the processed __init__input. For a one-element input a different approach to create the parameter is needed.

        Parameters:
            arg [str]: an __init__ input string that needs to be processed. The variables are separated by ','.
            level [int]: Level of differentiation of the returned function.

        Returns:
            matrix [Matrix]: a Matrix of dynamic symbols given by arg. 
        '''
        if arg is None:
            return None
        elif isinstance(arg, NDimArray):
            return arg
        else:
            if (',' in arg):
                return Array(dynamicsymbols(arg, level))
            else:
                return Array([dynamicsymbols(arg, level)])


    def createVariables(self, input_diffs:bool=False) -> tuple:
        """
        Returns a tuple with all variables. First the states are given, next the derivative of the states, and finally the inputs, optionally followed by the diffs of the inputs. All variables are sympy dynamic symbols.

        Parameters:
        -----------
            input_diffs : boolean
                also return the differentiated versions of the inputs, default: false.

        Returns:
        --------
            variables : tuple
                all variables of the system.

        Examples:
            * Return the variables of `sys', which has two states and two inputs and add a system to the SytemBase object:
            >>> from sympy.tensor.array import Array
            >>> from simupy.systems.symbolic import DynamicalSystem
            >>> x1, x2, x1dot, x2dot, u1, u2, u1dot, u2dot = sys.createVariables(input_diffs=True)
            >>> state_eq = Array([-5 * x1 + x2 + u1**2, x1/2 - x2**3 + u2])
            >>> output_eq = Array([x1 + x2])
            >>> sys.system = DynamicalSystem(input_=Array([u1, u2], state=Array([x1, x2], state_equation=state_eq, output_equation=output_eq)
        """
        if self.states is None:
            if len(tuple(self.inputs)) == 1:
                return tuple(self.inputs)[0]
            else:
                return tuple(self.inputs)
        else:
            states_matrix = Matrix(self.states)
            dstates_matrix = Matrix(self.dstates)
            inputs_matrix = Matrix(self.inputs)
            
            var_list_states = dstates_matrix.row_insert(0, states_matrix)
            var_list = inputs_matrix.row_insert(0, var_list_states)
            if input_diffs:
                input_diff_list = Matrix([diff(input_el, Symbol('t')) for input_el in inputs_matrix])
                var_list = input_diff_list.row_insert(0, var_list)
            return tuple(var_list)


    def series(self, sys_append):
        """
            A system is generated which is the result of a serial connection of two systems. The outputs of this object are connected to the inputs of the appended system and a new system is achieved which has the inputs of the current system and the outputs of the appended system. Notice that the dimensions of the output of the current system should be equal to the dimension of the input of the appended system.

            Parameters:
            -----------
                sys_append : SystemBase object
                    the system that is placed in a serial configuration. `sys_append' follows the current system.

            Returns:
            --------
                A SystemBase object with the serial system's equations.

            Examples:
            ---------
                * Place `sys1' behind `sys2' in a serial configuration and show the inputs, states, state equations and output equations:
                >>> series_sys = sys1.series(sys2)
                >>> print('inputs: ', series_sys.system.input_)
                >>> print('States: ', series_sys.system.state)
                >>> print('State eq's: ', series_sys.system.state_equation)
                >>> print('Output eq's: ', series_sys.system.output_equation)
        """
        if (self.sys.dim_output != sys_append.sys.dim_input):
            error_text = '[SystemBase.series] Dimension of output of the first system is not equal to dimension of input of the second system.'
            raise ValueError(error_text)
            # raise SystemExit(error_text), None, sys.exc_info()[2]
        else:
            inputs = self.inputs
            substitutions = dict(zip(sys_append.sys.input, self.sys.output_equation))
            output_equations =  Array([msubs(expr, substitutions) for expr in sys_append.sys.output_equation])
            if (self.states is None):
                if (sys_append.states is None):
                    return SystemBase(None, inputs, MemorylessSystem(input_=inputs, output_equation=output_equations))
                else:
                    states = sys_append.states
                    state_equations = Array([msubs(expr, substitutions) for expr in sys_append.sys.state_equation])
                    return SystemBase(states, inputs, DynamicalSystem(state_equation=state_equations, state=states, input_=inputs, output_equation=output_equations))
            else:
                if (sys_append.states is None):
                    states = self.states
                    state_equations = self.sys.state_equation
                else:
                    states = Array(self.states.tolist() + sys_append.states.tolist())
                    state_equations2 = Array(msubs(expr, substitutions) for expr in sys_append.sys.state_equation)
                    state_equations = Array(self.sys.state_equation.tolist() + state_equations2.tolist())
                return SystemBase(states, inputs, DynamicalSystem(state_equation=state_equations, state=states, input_=inputs, output_equation=output_equations))


    def parallel(self, sys_append):
        """
            A system is generated which is the result of a parallel connection of two systems. The inputs of this object are connected to the system that is placed in parallel and a new system is achieved with the output the sum of the outputs of both systems in parallel. Notice that the dimensions of the inputs and the outputs of both systems should be equal.

            Parameters:
            -----------
                sys_append : SystemBase object
                    the system that is added in parallel.

            Returns:
            --------
                A SystemBase object with the parallel system's equations.

            Examples:
            ---------
                * Place `sys2' in parallel with `sys1' and show the inputs, states, state equations and output equations:
                >>> parallel_sys = sys1.parallel(sys2)
                >>> print('inputs: ', parallel_sys.system.input_)
                >>> print('States: ', parallel_sys.system.state)
                >>> print('State eq's: ', parallel_sys.system.state_equation)
                >>> print('Output eq's: ', parallel_sys.system.output_equation)
        """
        if (self.sys.dim_input != sys_append.sys.dim_input):
            error_text = '[SystemBase.parallel] Dimension of the input of the first system is not equal to the dimension of the input of the second system.'
            raise ValueError(error_text)
        elif (self.sys.dim_output != sys_append.sys.dim_output):
            error_text = '[SystemBase.parallel] Dimension of the output of the first system is not equal to the dimension of the output of the second system.'
            raise ValueError(error_text)
        else:
            inputs = self.inputs
            substitutions = dict(zip(sys_append.sys.input, self.sys.input))
            output_equations = Array([value[0] + value[1] for value in zip(self.sys.output_equation, [msubs(expr, substitutions) for expr in sys_append.sys.output_equation])])
            if (self.states is None):
                if (sys_append.states is None):
                    return SystemBase(None, inputs, MemorylessSystem(input_=inputs, output_equation=output_equations))
                else:
                    states = sys_append.states
                    state_equations = Array([msubs(expr, substitutions) for expr in sys_append.sys.state_equation])
                    return SystemBase(states, inputs, DynamicalSystem(state_equation=state_equations, state=states, input_=inputs, output_equation=output_equations))
            else:
                if (sys_append.states is None):
                    states = self.states
                    state_equations = self.sys.state_equation
                else:
                    states = Array(self.states.tolist() + sys_append.states.tolist())
                    state_equations2 = Array(msubs(expr, substitutions) for expr in sys_append.sys.state_equation)
                    state_equations = Array(self.sys.state_equation.tolist() + state_equations2.tolist())
                return SystemBase(states, inputs, DynamicalSystem(state_equation=state_equations, state=states, input_=inputs, output_equation=output_equations))

    
    def simulation(self, tspan, number_of_samples=100, initial_conditions=None, input_signals=None, plot=False, custom_integrator_options=None):
        """
        Simulates the system in various conditions. It is possible to impose initial conditions on the states of the system. A specific input signal can be applied to the system to check its behavior. The results of the simulation are numerically available. Also, a plot of the states, inputs, and outputs is available. To simulate the system scipy's ode is used if the system has states. Both the option of variable time-step and fixed time step are available. If there are no states, a time signal is applied to the system.
        # TODO: output_signal -> a disturbance on the output signal.

        Parameters:
        -----------
            tspan : float or list-like
                the parameter defines the time vector for the simulation in seconds. An integer indicates the end time. A list-like object with two elements indicates the start and end time respectively. And more than two elements indicates at which time instances the system needs to be simulated.
            number_of_samples : int, optional
                number of samples in the case that the system is stateless and tspan only indicates the end and/or start time (span is length two or smaller), default: 100
            initial_conditions : int, float, list-like object, optional
                the initial conditions of the states of a statefull system. If none is given, all are zero, default: None
            input_signals : SystemBase object
                the input signal that is directly connected to the system's inputs. Preferably, the signals in nlcontrol.signals are used. If no input signal is specified and the system has inputs, all inputs are defaulted to zero, default: None
            plot : boolean, optional
                the plot boolean decides whether to show a plot of the inputs, states, and outputs, default: False
            custom_integrator_options : dict, optional (default: None)
                Specify specific integrator options top pass to ``integrator_class.set_integrator (scipy ode). The options are 'name', 'rtol', 'atol', 'nsteps', and 'max_step', which specify the integrator name, relative tolerance, absolute tolerance, number of steps, and maximal step size respectively. If no custom integrator options are specified the DEFAULT_INTEGRATOR_OPTIONS are used:
                    {
                        'name': 'dopri5',
                        'rtol': 1e-6,
                        'atol': 1e-12,
                        'nsteps': 500,
                        'max_step': 0.0
                    }
        Returns:
        --------
            A tuple:
                -> statefull system : 
                    t : ndarray
                        time vector.
                    x : ndarray
                        state vectors.
                    y : ndarray
                        input and ouput vectors.
                    res : SimulationResult object
                        A class object which contains information on events, next to the above vectors.
                -> stateless system :
                    t : ndarray
                        time vector.
                    y : ndarray
                        output vectors.
                    u : ndarray
                        input vectors. Is an empty list if the system has no inputs.

        Examples:
        ---------
            * A simulation of 20 seconds of the statefull system `sys' for a set of initial conditions [x0_0, x1_0, x2_0] and plot the results:
            >>> init_cond = [0.3, 5.7, 2]
            >>> t, x, y, res = sys.simulation(20, initial_conditions=init_cond)

            * A simulation from second 2 to 18 of the statefull system `sys' for an input signal, which is a step from 0.4 to 1.3 at second 5 for input 1 and from 0.9 to 1.1 at second 7. Use 1000 nsteps for the integrator. No plot is required:
            >>> from nlcontrol.signals import step
            >>> step_signal = step(step_times=[5, 7], begin_values=[0.4, 0.9], end_values=[1.3, 11])
            >>> integrator_options = {'nsteps': 1000}
            >>> t, x, y, res = sys.simulation([2, 18], input_signals=step_signal, custom_integrator_options=integrator_options)

            * Plot the stateless signal step from previous example for a custom time axis (a time axis going from 3 seconds to 20 seconds with 1000 equidistant samples in between):
            >>> import numpy as np
            >>> time_axis = np.linspace(3, 20, 1000)
            >>> t, y, _ = step_signal.simulation(time_axis, plot=True)
            Or
            >>> t, y, _ = step_signal.simulation([3, 20], number_of_samples=1000, plot=True)

            * Simulate the stateless system `sys_stateless' with input signal step_signal from the previous examples for 40 seconds with 1500 samples in between and plot:
            >>> t, y, u = sys_stateless.simulation(40, number_of_samples=1500, input_signals=step_signal, plot=True)
        """
        base_system = self.__copy__()
        if base_system.states is None:
            if np.isscalar(tspan):
                t = np.linspace(0, tspan, number_of_samples)
            elif len(tspan) == 2:
                t = np.linspace(tspan[0], tspan[1], number_of_samples)
            else:
                t = np.array(tspan)

            func = self.__get_output_equation()
            if base_system.inputs is None:
                u = []
                y = np.stack([func(t_el) for t_el in t])
            else:
                if input_signals is None:
                    input_signals = sgnls.empty_signal(base_system.system.dim_input)
                u = np.stack([input_signals.system.output_equation_function(t_i) for t_i in t])
                y = np.stack([func(u_el) for u_el in u])
            if plot:
                plt.figure()
                for k in range(base_system.system.dim_output):
                    plt.plot(t, y[:, k], label='y' + str(k))
                if not (base_system.inputs is None):
                    for l in range(base_system.system.dim_input):
                        plt.plot(t, u[:, l], label='u' + str(l))
                plt.title('inputs and outputs versus time')
                plt.xlabel('time (s)')
                plt.legend()
                plt.show()
            return t, y, u
            
        else: 
            BD = self.__connect_input(input_signals, base_system=base_system)
            if initial_conditions is not None:
                if isinstance(initial_conditions, (int, float)):
                    initial_conditions = [initial_conditions]
                base_system.system.initial_condition = initial_conditions

            max_inputs_index = base_system.system.dim_input
            if custom_integrator_options is not None:
                for key, _ in custom_integrator_options.items():
                    if not (key in ("name", "rtol", "atol", "nsteps", "max_step")):
                        error_text = "[SystemBase.simulation] the custom_integrator_options accepts the keywords name, rtol, atol, nsteps, and max_step. The keyword {} is not recognized.".format(key)
                        raise KeyError(error_text)

                integrator_options = {
                    'name': custom_integrator_options['name'] if 'name' in custom_integrator_options else 'dopri5',
                    'rtol': custom_integrator_options['rtol'] if 'rtol' in custom_integrator_options else 1e-6,
                    'atol': custom_integrator_options['atol'] if 'atol' in custom_integrator_options else 1e-12,
                    'nsteps': custom_integrator_options['nsteps'] if 'nsteps' in custom_integrator_options else 500,
                    'max_step': custom_integrator_options['max_step'] if 'max_step' in custom_integrator_options else 0.0
                }
            else:
                integrator_options = DEFAULT_INTEGRATOR_OPTIONS
            if np.isscalar(tspan):
                res = BD.simulate(tspan, integrator_options=integrator_options)
            elif len(tspan) == 2:
                res = BD.simulate(tspan, integrator_options=integrator_options)
            else:
                res = self.__simulation_loop(tspan, BD, base_system.system, integrator_options)
            
            t = res.t
            x = res.x
            if len(res.y[0]) == (base_system.system.dim_input + base_system.system.dim_output):
                y = res.y[:, max_inputs_index:]
                u = res.y[:, :max_inputs_index]
            else:
                y = res.y
                shape_u = (len(t), base_system.system.dim_input)
                u = np.zeros(shape_u)

            if plot:
                plt.figure()
                plt.subplot(121)
                for i in range(base_system.system.dim_state):
                    plt.plot(t, x[:, i], label=base_system.states[i])
                plt.title('states versus time')
                plt.xlabel('time (s)')
                plt.legend()
                plt.subplot(122)
                for j in range(max_inputs_index):
                    plt.plot(t, u[:, j], label=base_system.inputs[j])
                for k in range(base_system.system.dim_output):
                    plt.plot(t, y[:, k], label='y' + str(k))
                plt.title('inputs and outputs versus time')
                plt.xlabel('time (s)')
                plt.legend()
                plt.show()
        
            return t, x, y, u, res


    def __connect_input(self, input_signal=None, base_system=None, block_diagram=None):
        """
        Connects an input signal to a SystemBase object in a new simupy.BlockDiagram or in an existing simupy.BlockDiagram.
        """
        if base_system is None:
            base_system = self
        elif not isinstance(base_system, SystemBase):
            error_text = '[SystemBase.__connect_input] The system should be an SystemBase instance.'
            raise TypeError(error_text)

        if input_signal is None:
            if block_diagram is None:
                BD = BlockDiagram(base_system.system)
            elif not base_system.system in set(block_diagram.systems):
                    BD = block_diagram
                    BD.add_system(base_system.system)
        else:
            if block_diagram is None:
                BD = BlockDiagram(input_signal.system, base_system.system)
            else:
                BD = block_diagram
                BD.add_system(input_signal.system)
                if not base_system.system in set(block_diagram.systems):
                    BD.add_system(base_system.system)
            
            BD.connect(input_signal.system, base_system.system)
        return BD


    def __simulation_loop(self, time, block_diagram, system_with_states, integrator_options):
        """
        Loop through a time vector and simulate the simupy.BlockDiagram object for a each given time. Returns a simupy.SimulationResult object.
        """
        res = SimulationResult(block_diagram.cum_states[-1], block_diagram.cum_outputs[-1],time, block_diagram.systems.size)
        for index, t in enumerate(time):
            if index == 0:
                tspan = [0, t]
            else:
                tspan = [time[index - 1], t]
            res_temp = block_diagram.simulate(tspan, integrator_options=integrator_options)
            # print(index, ': ', tspan, ' - ', res_temp.t)
            res.t[index] = res_temp.t[-1]
            res.x[index] = res_temp.x[-1]
            res.y[index] = res_temp.y[-1]
            res.e[index] = res_temp.e[-1]
            system_with_states.initial_condition = res.x[index]
        return res


    def __get_output_equation(self):
        """
        Returns the lambdified output equation function of a SystemBase object.
        """
        if self.inputs is None:
            if (isinstance(self.system, DynamicalSystem2)):
                return self.system.output_equation_function
            else:
                error_text = '[system.__get_output_equation] The datatype DynamicalSystem2 is expected and not DynamicalSystem.'
                raise TypeError(error_text)
        else:
            if (isinstance(self.system, DynamicalSystem)):
                return lambdify_with_vector_args(self.system.input, self.system.output_equation)
            else:
                error_text = '[system.__get_output_equation] The datatype DynamicalSystem is expected and not DynamicalSystem2.'
                raise TypeError(error_text)
            

    
    def linearize(self, working_point:list):
        """
        A nonlinear system is linearized around a working point.

        Parameters:
        -----------
            working_point : list or int
                the x_dot is linearized around the working point.

        Returns:
        --------

        """
        pass


        