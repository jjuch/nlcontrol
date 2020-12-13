import nlcontrol.signals as sgnls

from copy import deepcopy, copy
import warnings
import types

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
import control

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

    .. math::
        \\frac{dx(t)}{dt} = h(x(t), u(t), t)

    with x(t) the state vector, u(t) the input vector and t the time in seconds. Next, the output is given by the output equation:
    
    .. math::
        y(t) = g(x(t), u(t), t)

    A SystemBase object contains several basic functions to manipulate and simulate the system.

    Parameters
    -----------
    states : string or array-like
        if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols.
    inputs : string or array-like
        if `inputs` is a string, it is a comma-separated listing of the input names. If `inputs` is array-like it contains the inputs as sympy's dynamic symbols.
    system : simupy's DynamicalSystem object (simupy.systems.symbolic), optional
        the object containing output and state equations, default: None.

    Examples
    ---------
    * Statefull system with one state, one input, and one output:
        >>> from simupy.systems.symbolic import MemorylessSystem, DynamicalSystem
        >>> from sympy.tensor.array import Array
        >>> states = 'x'
        >>> inputs = 'u'
        >>> sys = SystemBase(states, inputs)
        >>> x, xdot, u = sys.create_variables()
        >>> sys.system = DynamicalSystem(state_equation=Array([-x + u1]), state=x, output_equation=x, input_=u1)

    * Statefull system with two states, one input, and two outputs:
        >>> states = 'x1, x2'
        >>> inputs = 'u'
        >>> sys = SystemBase(states, inputs)
        >>> x1, x2, x1dot, x2dot, u = sys.create_variables()
        >>> sys.system = DynamicalSystem(state_equation=Array([-x1 + x2**2 + u, -x2 + 0.5 * x1]), state=Array([x1, x2]), output_equation=Array([x1 * x2, x2]), input_=u)

    * Stateless system with one input:
        >>> states = None
        >>> inputs = 'w'
        >>> sys = SystemBase(states, inputs)
        >>> w = sys.create_variables()
        >>> sys.system = MemorylessSystem(input_=Array([w]), output_equation= Array([5 * w]))

    * Create a copy a SystemBase object `sys' and linearize around the working point of state [0, 0] and working point of input 0 and simulate:
        >>> new_sys = SystemBase(sys.states, sys.inputs, sys.system)
        >>> new_sys_lin = new_sys.linearize([0, 0], 0)
        >>> new_sys_lin.simulation(10)

    """
    def __init__(self, states, inputs, sys=None):
        self.states = self.__process_init_input__(states)
        self.dstates = self.__process_init_input__(states, 1)
        self.inputs = self.__process_init_input__(inputs)
        self.sys = sys


    def __str__(self):
        if callable(self.output_equation):
            try:
                output_equation = str(self.output_equation(Symbol('t')))
            except:
                output_equation = "callable(t)"
        else:
            output_equation = str(self.output_equation)

        if len(output_equation) > 50:
            output_equation = "\n\t\t\t{}\n".format(str(output_equation).replace(",", ",\n\t\t\t"))
        
        if self.state_equation is not None and (len(str(self.state_equation)) > 50):
            state_equation = "\n\t\t\t{}\n".format(str(self.state_equation).replace(",", ",\n\t\t\t"))
        else:
            state_equation = str(self.state_equation)


        return """
        SystemBase object:
        ==================
        Inputs: {}\n
        States: {}\n
        System: 
        \tState eq.: {}
        \tOutput eq.: {}
        """.format(self.inputs, self.states, state_equation, output_equation)
        

    def __copy__(self):
        """
        Create a deep copy of the SystemBase object.
        """
        return deepcopy(self)

    @property
    def system(self):
        """
        :obj:`simupy's DynamicalSystem`
        
        The system attribute of the SystemBase class. The system is defined using `simupy's DynamicalSystem <https://simupy.readthedocs.io/en/latest/api/symbolic_systems.html#simupy.systems.symbolic.DynamicalSystem>`__.
        """
        return self.sys

    @system.setter
    def system(self, system):
        self.sys = system

    @property
    def state_equation(self):
        """
        :obj:`expression` containing :obj:`dynamicsymbols`
        
        The state equation contains `sympy's dynamicsymbols <https://docs.sympy.org/latest/modules/physics/vector/api/functions.html#dynamicsymbols>`__.
        """
        if self.states is not None:
            return self.system.state_equation

    @property
    def output_equation(self):
        """
        :obj:`expression` containing :obj:`dynamicsymbols`
        
        The output equation contains `sympy's dynamicsymbols <https://docs.sympy.org/latest/modules/physics/vector/api/functions.html#dynamicsymbols>`__.
        """
        if hasattr(self.system, 'output_equation'):
            return self.system.output_equation
        elif hasattr(self.system, 'output_equation_function'):
            return self.system.output_equation_function


    @property
    def block_configuration(self):
        """
        Returns info on the systems: the dimension of the inputs, the states, and the output. This property is mainly intended for debugging.
        """

        sys = self.system
        print("""
        Inputs: {}
        States: {}
        Outputs: {}
        """.format(sys.dim_input, sys.dim_state, sys.dim_output))


    def __process_init_input__(self, arg:str, level:int=0) -> Matrix:
        """
        Return the correct format of the processed __init__input. For a one-element input a different approach to create the parameter is needed.

        Parameters
        -----------
        arg : string or array-like
            an __init__ input string that needs to be processed. The variables are separated by ','. Or an NDimArray object, which is returned without any adaptations.
        level : int
            Level of differentiation of the returned function.

        Returns
        --------
        matrix [Matrix]: a Matrix of dynamic symbols given by arg. 
        """
        if arg is None:
            return None
        elif isinstance(arg, NDimArray):
            if level == 0:
                return arg
            else:
                return Array([diff(st, Symbol('t'), level) for st in arg])
        else:
            if (',' in arg):
                return Array(dynamicsymbols(arg, level))
            else:
                return Array([dynamicsymbols(arg, level)])


    def create_variables(self, input_diffs:bool=False, states=None) -> tuple:
        """
        Returns a tuple with all variables. First the states are given, next the derivative of the states, and finally the inputs, optionally followed by the diffs of the inputs. All variables are sympy dynamic symbols.

        Parameters
        -----------
        input_diffs : boolean
            also return the differentiated versions of the inputs, default: false.
        states : array-like
            An alternative list of states, used by more complex system models, optional. (see e.g. EulerLagrange.create_variables)

        Returns
        --------
        variables : tuple
            all variables of the system.

        Examples
        --------
        * Return the variables of `sys', which has two states and two inputs and add a system to the SytemBase object:
        >>> from sympy.tensor.array import Array
        >>> from simupy.systems.symbolic import DynamicalSystem
        >>> x1, x2, x1dot, x2dot, u1, u2, u1dot, u2dot = sys.create_variables(input_diffs=True)
        >>> state_eq = Array([-5 * x1 + x2 + u1**2, x1/2 - x2**3 + u2])
        >>> output_eq = Array([x1 + x2])
        >>> sys.system = DynamicalSystem(input_=Array([u1, u2], state=Array([x1, x2], state_equation=state_eq, output_equation=output_eq)
        """
        if states is None:
            states = self.states
            dstates = self.dstates
        else:
            dstates = [diff(state, Symbol('t')) for state in states]
        if states is None:
            # if len(tuple(self.inputs)) == 1:
                
            #     return tuple(self.inputs)[0]
            # else:
            #     return tuple(self.inputs)
            inputs_matrix = Matrix(self.inputs)
            if input_diffs:
                input_diff_list = Matrix([diff(input_el, Symbol('t')) for input_el in inputs_matrix])
                var_list = input_diff_list.row_insert(0, inputs_matrix)
            else:
                var_list = inputs_matrix
            return tuple(var_list) if len(var_list) > 1 else var_list
                
        else:
            states_matrix = Matrix(states)
            dstates_matrix = Matrix(dstates)
            inputs_matrix = Matrix(self.inputs)
            
            var_list_states = dstates_matrix.row_insert(0, states_matrix)
            var_list = inputs_matrix.row_insert(0, var_list_states)
            if input_diffs:
                input_diff_list = Matrix([diff(input_el, Symbol('t')) for input_el in inputs_matrix])
                var_list = input_diff_list.row_insert(0, var_list)
            return tuple(var_list) if len(var_list) > 1 else var_list

    
    def linearize(self, working_point_states, working_point_inputs=None):
        """
        In many cases a nonlinear system is observed around a certain working point. In the state space close to this working point it is save to say that a linearized version of the nonlinear system is a sufficient approximation. The linearized model allows the user to use linear control techniques to examine the nonlinear system close to this working point. A first order Taylor expansion is used to obtain the linearized system. A working point for the states is necessary, but the working point for the input is optional.

        Parameters
        -----------
        working_point_states : list or int
            the state equations are linearized around the working point of the states.
        working_point_inputs : list or int
            the state equations are linearized around the working point of the states and inputs.

        Returns
        --------
        sys_lin: SystemBase object 
            with the same states and inputs as the original system. The state and output equation is linearized.
        sys_control: control.StateSpace object

        Examples
        ---------
        * Print the state equation of the linearized system of `sys' around the state's working point x[1] = 1 and x[2] = 5 and the input's working point u = 2:
            >>> sys_lin, sys_control = sys.linearize([1, 5], 2)
            >>> print('Linearized state equation: ', sys_lin.state_equation)
            

        """
        if type(self.system) is DynamicalSystem2:
            warnings.warn("[SystemBase.linearize] A dynamical system with a lambdafied state equation cannot be linearized.", UserWarning)
            return None
        if np.isscalar(working_point_states):
            working_point_states = [working_point_states]
        if (len(working_point_states) != len(self.states)):
            error_text = '[SystemBase.linearize] The working point should have the same size as the dimension of the states. The dimension of the state is {}.'.format(len(self.states))
            raise ValueError(error_text)

        substitutions_states = dict(zip(self.states, working_point_states))
        if working_point_inputs is not None:
            if np.isscalar(working_point_inputs):
                working_point_inputs = [working_point_inputs]
            if (len(working_point_inputs) != len(self.inputs)):
                error_text = '[SystemBase.linearize] The working point should have the same size as the dimension of the inputs.'
                raise ValueError(error_text)

            substitutions_inputs = dict(zip(self.inputs, working_point_inputs))
            substitutions_states = dict(list(substitutions_states.items()) + list(substitutions_inputs.items()))

        def create_linear_equation(nl_expr):
            linearized_expr = []
            # for k in range(len(self.dstates)):
            for k in range(len(nl_expr)):
                linearized_term = 0
                for j in range(len(self.states)):
                    linearized_term += msubs(diff(nl_expr[k], self.states[j]), substitutions_states) * (self.states[j] - substitutions_states[self.states[j]])

                if working_point_inputs is not None:
                    for l in range(len(self.inputs)):
                        linearized_term += msubs(diff(nl_expr[k], self.inputs[l]), substitutions_states) * (self.inputs[l] - substitutions_states[self.inputs[l]])
                linearized_term += msubs(nl_expr[k], substitutions_states)
                linearized_expr.append(linearized_term)
            return Array(linearized_expr)

        state_equation_linearized = create_linear_equation(self.system.state_equation)
        output_equation_linearized = create_linear_equation(self.system.output_equation)
        system_dyn = DynamicalSystem(state_equation=state_equation_linearized, state=self.states, input_=self.inputs, output_equation=output_equation_linearized)
        system = SystemBase(states=self.states, inputs=self.inputs, sys=system_dyn)
        
        def get_state_space_matrices(state_equations, output_equations):
            A = []
            B = []
            C = []
            D = []
            for i in range(len(state_equations)):
                col_A = [state_equations[i].coeff(state) for state in self.states]
                A.append(col_A)
                col_B = [state_equations[i].coeff(input_el) for input_el in self.inputs] if self.inputs is not None else [0]
                B.append(col_B)
            for j in range(len(output_equations)):
                col_C = [output_equations[j].coeff(state) for state in self.states]
                C.append(col_C)
                col_D = [output_equations[j].coeff(input_el) for input_el in self.inputs] if self.inputs is not None else [0]
                D.append(col_D)
            return np.array(A), np.array(B), np.array(C), np.array(D)

        A, B, C, D = get_state_space_matrices(state_equation_linearized, output_equation_linearized)
        sys_control = control.ss(A, B, C, D)
        return system, sys_control



    def series(self, sys_append):
        """
        A system is generated which is the result of a serial connection of two systems. The outputs of this object are connected to the inputs of the appended system and a new system is achieved which has the inputs of the current system and the outputs of the appended system. Notice that the dimensions of the output of the current system should be equal to the dimension of the input of the appended system.

        Parameters
        -----------
        sys_append : SystemBase object
            the system that is placed in a serial configuration. 'sys_append' follows the current system.

        Returns
        --------
        A SystemBase object with the serial system's equations.

        Examples
        ---------
        * Place 'sys1' behind 'sys2' in a serial configuration and show the inputs, states, state equations and output equations:
            >>> series_sys = sys1.series(sys2)
            >>> print('inputs: ', series_sys.system.input_)
            >>> print('States: ', series_sys.system.state)
            >>> print('State eqs: ', series_sys.system.state_equation)
            >>> print('Output eqs: ', series_sys.system.output_equation)
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

        Parameters
        -----------
        sys_append : SystemBase object
            the system that is added in parallel.

        Returns
        --------
        A SystemBase object with the parallel system's equations.

        Examples
        ---------
        * Place 'sys2' in parallel with 'sys1' and show the inputs, states, state equations and output equations:
            >>> parallel_sys = sys1.parallel(sys2)
            >>> print('inputs: ', parallel_sys.system.input_)
            >>> print('States: ', parallel_sys.system.state)
            >>> print('State eqs: ', parallel_sys.system.state_equation)
            >>> print('Output eqs: ', parallel_sys.system.output_equation)
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

        Parameters
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
            Specify specific integrator options top pass to integrator_class.set_integrator (scipy ode)`. The options are 'name', 'rtol', 'atol', 'nsteps', and 'max_step', which specify the integrator name, relative tolerance, absolute tolerance, number of steps, and maximal step size respectively. If no custom integrator options are specified the ``DEFAULT_INTEGRATOR_OPTIONS`` are used:

            .. code-block:: json

                {
                    'name': 'dopri5',
                    'rtol': 1e-6,
                    'atol': 1e-12,
                    'nsteps': 500,
                    'max_step': 0.0
                }


        Returns
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

        Examples
        ---------
        * A simulation of 20 seconds of the statefull system 'sys' for a set of initial conditions [x0_0, x1_0, x2_0] and plot the results:
            >>> init_cond = [0.3, 5.7, 2]
            >>> t, x, y, u, res = sys.simulation(20, initial_conditions=init_cond)

        * A simulation from second 2 to 18 of the statefull system 'sys' for an input signal, which is a step from 0.4 to 1.3 at second 5 for input 1 and from 0.9 to 1.1 at second 7. Use 1000 nsteps for the integrator. No plot is required:
            >>> from nlcontrol.signals import step
            >>> step_signal = step(step_times=[5, 7], begin_values=[0.4, 0.9], end_values=[1.3, 11])
            >>> integrator_options = {'nsteps': 1000}
            >>> t, x, y, u, res = sys.simulation([2, 18], input_signals=step_signal, custom_integrator_options=integrator_options)

        * Plot the stateless signal step from previous example for a custom time axis (a time axis going from 3 seconds to 20 seconds with 1000 equidistant samples in between):
            >>> import numpy as np
            >>> time_axis = np.linspace(3, 20, 1000)
            >>> t, y, _ = step_signal.simulation(time_axis, plot=True)
            Or
            >>> t, y, _ = step_signal.simulation([3, 20], number_of_samples=1000, plot=True)

        * Simulate the stateless system 'sys_stateless' with input signal step_signal from the previous examples for 40 seconds with 1500 samples in between and plot:
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

            func = self.__get_output_equation__()
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
            BD = self.__connect_input__(input_signals, base_system=base_system)
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
                res = self.__simulation_loop__(tspan, BD, base_system.system, integrator_options)
            
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


    def __connect_input__(self, input_signal=None, base_system=None, block_diagram=None):
        """
        Connects an input signal to a SystemBase object in a new simupy.BlockDiagram or in an existing simupy.BlockDiagram.
        """
        if base_system is None:
            base_system = self
        elif not isinstance(base_system, SystemBase):
            error_text = '[SystemBase.__connect_input__] The system should be an SystemBase instance.'
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


    def __simulation_loop__(self, time, block_diagram, system_with_states, integrator_options):
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


    def __get_output_equation__(self):
        """
        Returns the lambdified output equation function of a SystemBase object.
        """
        if self.inputs is None:
            if (isinstance(self.system, DynamicalSystem2)):
                return self.system.output_equation_function
            else:
                error_text = '[system.__get_output_equation__] The datatype DynamicalSystem2 is expected and not DynamicalSystem.'
                raise TypeError(error_text)
        else:
            if (isinstance(self.system, DynamicalSystem)):
                return lambdify_with_vector_args(self.system.input, self.system.output_equation)
            else:
                error_text = '[system.__get_output_equation__] The datatype DynamicalSystem is expected and not DynamicalSystem2.'
                raise TypeError(error_text)


class TransferFunction(SystemBase):
    def __init__(self, *args):
        sys = control.tf2ss(*args)
        print(sys)
