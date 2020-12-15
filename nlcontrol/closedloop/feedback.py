from simupy.block_diagram import BlockDiagram
from simupy.systems.symbolic import DynamicalSystem
from sympy.matrices import Matrix
from sympy.physics.mechanics import msubs
from sympy.tensor.array import Array

from nlcontrol.closedloop.blocks import gain_block
from nlcontrol.systems import SystemBase, ControllerBase
from nlcontrol.visualisation import ClosedLoopRenderer

import numpy as np
import matplotlib.pyplot as plt

__all__ = ["ClosedLoop"]


class ClosedLoop(object):
    """
    ClosedLoop(forward=None, backward=None)

    The object contains a closed loop configuration using BlockDiagram objects of the simupy module. The closed loop systems is given by the following block scheme:

    .. aafig::
        :aspect: 75
        :scale: 100
        :proportional:
        :textual:

                 u  +----------+    w
        +---------->+  Forward +------+---->
        |           +----------+      |
        |                             |     
        | +----+   +--------------+   |
        +-+ -1 +<--+   Backward   +<--+
          +----+   +--------------+
    

    Parameters
    -----------
    forward : :class:`Systembase` or :obj:`list` of :class:`Systembase`
        A state-full or state-less system. The number of inputs should be equal to the number of controller outputs.
    backward : :class:`Systembase` or :obj:`list` of :class:`Systembase`
        A state-full or state-less SystemBase object, which can also be a ControllerBase object. The number of inputs should be equal to the number of system outputs.

    Examples
    ---------
    * Create a closed-loop object of SystemBase object 'sys', which uses the Euler-Lagrange formulation, and ControllerBase object 'contr' containing a PID and a DynamicController object in parallel.
        >>> from nlcontrol import PID, DynamicController, EulerLagrange
        >>> $
        >>> # Define the system:
        >>> states = 'x1, x2'
        >>> inputs = 'u1, u2'
        >>> sys = EulerLagrange(states, inputs)
        >>> x1, x2, dx1, dx2, u1, u2, du1, du2 = sys.create_variables(input_diffs=True)
        >>> M = [[1, x1*x2],
            [x1*x2, 1]]
        >>> C = [[2*dx1, 1 + x1],
            [x2 - 2, 3*dx2]]
        >>> K = [x1, 2*x2]
        >>> F = [u1, 0]
        >>> sys.define_system(M, C, K, F)
        >>> $
        >>> # Define the DynamicController controller:
        >>> st = 'z1, z2'
        >>> dyn_contr = DynamicController(states=st, inputs=sys.minimal_states)
        >>> z1, z2, z1dot, z2dot, w, wdot = contr.create_variables()
        >>> a0, a1, k1 = 12.87, 6.63, 0.45
        >>> b0 = (48.65 - a1) * k1
        >>> b1 = (11.79 - 1) * k1
        >>> A = [[0, 1], [-a0, -a1]]
        >>> B = [[0], [1]]
        >>> C = [[b0], [b1]]
        >>> f = lambda x: x**2
        >>> eta = [[w + wdot], [(w + wdot)**2]]
        >>> phi = [[z1], [z2dot]]
        >>> contr.define_controller(A, B, C, f, eta, phi)
        >>> $
        >>> # Define the PID:
        >>> kp = 1
        >>> kd = 1
        >>> ksi0 = [kp * x1, kp * x2]
        >>> psi0 = [kd * dx1, kd * dx2]
        >>> pid = PID(ksi0, None, psi0, inputs=sys.minimal_states)
        >>> $
        >>> # Create the controller:
        >>> contr = dyn_contr.parallel(pid)
        >>> $
        >>> # Create a closed-loop object:
        >>> CL = ClosedLoop(sys, contr)
    """

    def __init__(self, forward=None, backward=None):
        self._fwd_system = None
        self._bwd_system = None
        self.forward_system = (forward, False)
        self.backward_system = (backward, False)
        self.block_diagram, self.indices = self.create_block_diagram()
        self.closed_loop_system = self.create_closed_loop_system()
        self.renderer = self.closed_loop_system.renderer

    @property
    def forward_system(self):
        """
        :class:`Systembase`

        The system in the forward path of the closed loop.
        """
        return self._fwd_system
    
    @forward_system.setter 
    def forward_system(self, system_args):
        # Parse input arguments
        if type(system_args) != tuple:
            system = system_args
            update_class_props = True
        elif len(system_args) == 2:
            system, update_class_props = system_args
        else:
            error_text="[ClosedLoop.forward_system] the function expects one or two arguments. {} were given.".format(len(system_args))
            raise TypeError(error_text)
        # Actual setter
        if issubclass(type(system), SystemBase):
            self._fwd_system = system
            if update_class_props:
                self.block_diagram, self.indices = self.create_block_diagram()
                self.closed_loop_system = self.create_closed_loop_system()
                self.renderer = self.closed_loop_system.renderer
        else:
            error_text = '[ClosedLoop.forward_system] the system object should be of the type SystemBase.'
            raise TypeError(error_text)
        

    @property
    def backward_system(self):
        """
        :obj:`SystemBase`

        The system (often a ControllerBase object) in the backward path of the closed loop.
        """
        return self._bwd_system
    
    @backward_system.setter 
    def backward_system(self, system_args):
        # Parse input arguments
        if type(system_args) != tuple:
            system = system_args
            update_class_props = True
        elif len(system_args) == 2:
            system, update_class_props = system_args
        else:
            error_text="[ClosedLoop.backward_system] the function expects one or two arguments. {} were given.".format(len(system_args))
            raise TypeError(error_text)
        # Actual setter
        if (issubclass(type(system), SystemBase)) or (system is None):
            self._bwd_system = system
            if update_class_props:
                self.block_diagram, self.indices = self.create_block_diagram()
                self.closed_loop_system = self.create_closed_loop_system()
                self.renderer = self.closed_loop_system.renderer
        else:
            error_text = '[ClosedLoop.backward_system] the controller object should be of the type ControllerBase.'
            raise TypeError(error_text)
        

    def create_closed_loop_system(self):
        '''
        Create a SystemBase object of the closed-loop system.

        Returns
        --------
        system : SystemBase
            A Systembase object of the closed-loop system.
        '''
        states, state_equations = self.__get_states__()
        # Create input vector
        input_dim = self.backward_system.system.dim_output
        inputs = SystemBase.__process_init_input__(None, 'r0:{}'.format(input_dim))

        # Define a simupy DynamicalSystem
        system_dyn = DynamicalSystem(state_equation=Array(state_equations), state=states, output_equation=self._fwd_system.output_equation, input_=inputs)
        return SystemBase(states=Array(states), inputs=inputs, sys=system_dyn, name="closed-loop", block_type='closedloop', forward_sys=self.forward_system, backward_sys=self.backward_system)

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
        \\ TODO
        """
        if (self.closed_loop_system.system.dim_output != sys_append.system.dim_input):
            error_text = '[ClosedLoop.series] Dimension of output of the closed-loop system is not equal to dimension of input of the appended system.'
            raise ValueError(error_text)
        else:
            return self.closed_loop_system.series(sys_append)


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
        \\TODO
        """
        if (self.closed_loop_system.system.dim_input != sys_append.system.dim_input):
            error_text = '[ClosedLoop.parallel] Dimension of the input of the closed-loop system is not equal to the dimension of the input of the appended system.'
            raise ValueError(error_text)
        elif (self.closed_loop_system.system.dim_output != sys_append.system.dim_output):
            error_text = '[ClosedLoop.parallel] Dimension of the output of the closed-loop system is not equal to the dimension of the output of the appended system.'
            raise ValueError(error_text)
        else:
            return self.closed_loop_system.parallel(sys_append)



    def __get_states__(self):
        '''
        Contcatenate the states vector of the system and the controller.

        Returns
        --------
        states : list
            first the states of the system and next the states of the controller.
        state_equations : list
            first the state equations of the system and next the state equations of the controller.
        '''
        states = []
        state_equations = []
        
        if self._fwd_system is None:
            if self._bwd_system is None:
                error_text = '[ClosedLoop.__get_states__] Both controller and system are None. One of them should at least contain a system.'
                raise AssertionError(error_text)
            else:
                if self._bwd_system.states is not None:
                    states.extend(self._bwd_system.states)
                    state_equations.extend(self._bwd_system.state_equation)
        else:
            substitutions_derivatives = dict()
            unprocessed_substitutions_system = zip(self._fwd_system.inputs, (-1) * self._bwd_system.output_equation)

            if self._fwd_system.states is not None:
                # Remove Derivative(., 't') from controller states and substitutions_system
                minimal_dstates = self._fwd_system.states[1::2]
                dstates = self._fwd_system.dstates[0::2]
                substitutions_derivatives = dict(zip(dstates, minimal_dstates))
                substitutions_system = dict([(k, msubs(v, substitutions_derivatives))\
                    for k, v in unprocessed_substitutions_system])

                states.extend(self._fwd_system.states)
                state_equations.extend([msubs(state_eq, substitutions_derivatives, substitutions_system)\
                    for state_eq in self._fwd_system.state_equation])                
                 
            if self._bwd_system.states is not None:
                states.extend(self._bwd_system.states)
                controller_state_eq = msubs(self._bwd_system.state_equation, substitutions_derivatives)
                state_equations.extend(controller_state_eq)     
        return states, state_equations
    

    def linearize(self, working_point_states):
        '''
        In many cases a nonlinear closed-loop system is observed around a certain working point. In the state space close to this working point it is save to say that a linearized version of the nonlinear system is a sufficient approximation. The linearized model allows the user to use linear control techniques to examine the nonlinear system close to this working point. A first order Taylor expansion is used to obtain the linearized system. A working point for the states needs to be provided.

        Parameters
        -----------
        working_point_states : list or int
            the state equations are linearized around the working point of the states.

        Returns
        --------
        sys_lin: SystemBase object 
            with the same states and inputs as the original system. The state and output equation is linearized.
        sys_control: control.StateSpace object

        Examples
        ---------
        * Print the state equation of the linearized closed-loop object of `CL' around the state's working point x[1] = 1 and x[2] = 5:
            >>> CL_lin, CL_control = CL.linearize([1, 5])
            >>> print('Linearized state equation: ', CL_lin.state_equation)
        '''
        return self.closed_loop_system.linearize(working_point_states)


    def create_block_diagram(self, forward_systems:list=None, backward_systems:list=None):
        """
        Create a closed loop block diagram with negative feedback. The loop contains a list of SystemBase objects in the forward path and ControllerBase objects in the backward path.

        Parameters
        -----------
        forward_systems : list, optional (at least one system should be present in the loop)
            A list of SystemBase objects. All input and output dimensions should match.
        backward_systems: list, optional (at least one system should be present in the loop)
            A list of ControllerBase objects. All input and output dimensions should match.

        Returns
        --------
        BD : a simupy's BlockDiagram object 
            contains the configuration of the closed-loop.
        indices : dict
            information on the ranges of the states and outputs in the output vectors of a simulation dataset.
        """
        if (forward_systems is None):
            if (self._fwd_system is None):
                error_text = "[ClosedLoop.create_block_diagram] Both the forward_systems argument and the ClosedLoop.system variable are empty. Please provide a forward_system."
                assert AssertionError(error_text)
            else:
                forward_systems = [self._fwd_system]
        if (backward_systems is None):
            if (self._fwd_system is None):
                error_text = "[ClosedLoop.create_block_diagram] Both the backward_systems argument and the ClosedLoop.controller variable are empty. Please provide a backward_system."
                assert AssertionError(error_text)
            else:
                backward_systems = [self._bwd_system]

        BD = BlockDiagram()
        # Order of adding systems is important. The negative feedback_block needs to be added before the backward systems. This can be seen from simupy.block_diagram.BlockDiagram.output_equation_function(). Possibly only for stateless systems important. #TODO: check whether there is a problem with list of backward systems.
        output_startidx_process = 0
        output_endidx_process = -1
        state_startidx_process = 0
        state_endidx_process = -1

        if (len(forward_systems) is not 0):
            for forward_system in forward_systems:
                forward_sys = forward_system.system
                BD.add_system(forward_sys)
                output_endidx_process += forward_sys.dim_output
                state_endidx_process += forward_sys.dim_state
        output_endidx_process += 1
        state_endidx_process += 1

        output_startidx_controller = output_endidx_process
        output_endidx_controller = output_startidx_controller
        state_startidx_controller = state_endidx_process
        state_endidx_controller = state_startidx_controller

        if (len(backward_systems) is not 0):
            negative_feedback = gain_block(-1, backward_systems[-1].system.dim_output)
            BD.add_system(negative_feedback)
            output_startidx_controller += negative_feedback.dim_output
            output_endidx_controller = output_startidx_controller
            for backward_system in backward_systems:
                backward_sys = backward_system.system
                BD.add_system(backward_sys)
                output_endidx_controller += backward_sys.dim_output
                state_endidx_controller += backward_sys.dim_state
        else:
            negative_feedback = gain_block(-1, forward_systems[-1].system.dim_output)
            BD.add_system(negative_feedback)

        for i in range(len(forward_systems)):
            if (i == len(forward_systems) - 1):
                BD.connect(forward_systems[i].system, backward_systems[0].system)
            else:
                BD.connect(forward_systems[i].system, forward_systems[i + 1].system)
        if (len(backward_systems) == 0):
            BD.add_system(negative_feedback)
            BD.connect(forward_systems[-1].system, negative_feedback)
            BD.connect(negative_feedback, forward_systems[0].system)
        else:
            for j in range(len(backward_systems)):
                if (j == len(backward_systems) - 1):
                    BD.connect(backward_systems[j].system, negative_feedback)
                    BD.connect(negative_feedback, forward_systems[0].system)
                else:
                    BD.connect(backward_systems[j].system, backward_systems[j + 1].system)
        
        indices = {
            'process': {
                'output': [output_endidx_process] if output_startidx_process == output_endidx_process\
                    else [output_startidx_process, output_endidx_process],
                'state': None if state_endidx_process is 0\
                    else[state_endidx_process] if state_startidx_process == state_endidx_process\
                    else [state_startidx_process, state_endidx_process]
            },
            'controller': {
                'output': [output_endidx_controller] if output_startidx_controller == output_endidx_controller\
                    else [output_startidx_controller, output_endidx_controller],
                'state': None if state_endidx_controller == state_endidx_process\
                    else[state_endidx_controller] if state_startidx_controller == state_endidx_controller\
                    else [state_startidx_controller, state_endidx_controller]
            }
        }
        return BD, indices


    def simulation(self, tspan, initial_conditions, plot=False, custom_integrator_options=None):
        """
        Simulates the closed-loop in various conditions. It is possible to impose initial conditions on the states of the system. The results of the simulation are numerically available. Also, a plot of the states and outputs is available. To simulate the system scipy's ode is used. 
        # TODO: output_signal -> a disturbance on the output signal.

        Parameters
        -----------
        tspan : float or list-like
            the parameter defines the time vector for the simulation in seconds. An integer indicates the end time. A list-like object with two elements indicates the start and end time respectively. And more than two elements indicates at which time instances the system needs to be simulated.
        initial_conditions : int, float, list-like object
            the initial conditions of the states of a statefull system. If none is given, all are zero, default: None
        plot : boolean, optional
            the plot boolean decides whether to show a plot of the inputs, states, and outputs, default: False
        custom_integrator_options : dict, optional (default: None)
            Specify specific integrator options to pass to `integrator_class.set_integrator (scipy ode)`. The options are 'name', 'rtol', 'atol', 'nsteps', and 'max_step', which specify the integrator name, relative tolerance, absolute tolerance, number of steps, and maximal step size respectively. If no custom integrator options are specified the ``DEFAULT_INTEGRATOR_OPTIONS`` are used:

            .. code-block:: json

                {
                    "name": "dopri5",
                    "rtol": 1e-6,
                    "atol": 1e-12,
                    "nsteps": 500,
                    "max_step": 0.0
                }
        
        
        Returns
        --------
        t : array
            time vector
        data : tuple
            four data vectors, the states and the outputs of the systems in the forward path and the states and outputs of the systems in the backward path.


        Examples
        ---------
        * A simulation of 5 seconds of the statefull SystemBase object 'sys' in the forward path and the statefull ControllerBase object `contr' in the backward path for a set of initial conditions [x0_0, x1_0] and plot the results:
            >>> CL = ClosedLoop(sys, contr)
            >>> t, data = CL.simulation(5, [x0_0, x1_0], custom_integrator_options={'nsteps': 1000}, plot=True)
            >>> (x_p, y_p, x_c, y_c) = data
            
        """
        if custom_integrator_options is not None:
            integrator_options = {
                'name': custom_integrator_options['name'] if 'name' in custom_integrator_options else 'dopri5',
                'rtol': custom_integrator_options['rtol'] if 'rtol' in custom_integrator_options else 1e-6,
                'atol': custom_integrator_options['atol'] if 'atol' in custom_integrator_options else 1e-12,
                'nsteps': custom_integrator_options['nsteps'] if 'nsteps' in custom_integrator_options else 500,
                'max_step': custom_integrator_options['max_step'] if 'max_step' in custom_integrator_options else 0.0
            }
        else:
            integrator_options = {
                'name': 'dopri5',
                'rtol': 1e-6,
                'atol': 1e-12,
                'nsteps': 500,
                'max_step': 0.0
            }

        self._fwd_system.system.initial_condition = initial_conditions
        res = self.block_diagram.simulate(tspan, integrator_options=integrator_options)
        
        # Unpack indices
        y_p_idx = self.indices['process']['output']
        x_p_idx = self.indices['process']['state']
        y_c_idx = self.indices['controller']['output']
        x_c_idx = self.indices['controller']['state']
        y_p = res.y[:, y_p_idx[0]] if len(y_p_idx) == 1\
            else res.y[:, slice(*y_p_idx)]
        x_p = None if x_p_idx is None\
            else res.x[:, x_p_idx[0]] if len(x_p_idx) == 1\
            else res.x[:, slice(*x_p_idx)]
        y_c = res.y[:, y_c_idx[0]] if len(y_c_idx) == 1\
            else res.y[:, slice(*y_c_idx)]
        x_c = None if x_c_idx is None\
            else res.x[:, x_c_idx[0]] if len(x_c_idx) == 1\
            else res.x[:, slice(*x_c_idx)]

    
        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            ObjectLines1A = plt.plot(res.t, y_p)
            ObjectLines1B = plt.plot(res.t, y_c)
            plt.legend(iter(ObjectLines1A + ObjectLines1B), ['y' + str(index) for index in range(1, len(y_p[0]) + 1)] + ['u' + str(index) for index in range(1, len(y_c[0]) + 1)])
            plt.title('Outputs')
            plt.xlabel('time [s]')

            plt.subplot(1, 2, 2)
            ObjectLines2A = []
            ObjectLines2B = []
            if x_p is not None:
                ObjectLines2A = plt.plot(res.t, x_p)
            if x_c is not None:
                ObjectLines2B = plt.plot(res.t, x_c)
            plt.legend(iter(ObjectLines2A + ObjectLines2B), ['x' + str(index) for index in ([] if x_p is None else range(1, len(x_p[0]) + 1))] + ['z' + str(index) for index in ([] if x_c is None else range(1, len(x_c[0]) + 1))])
            plt.title('States')
            plt.xlabel('time [s]')
            plt.show()

        return res.t, (x_p, y_p, x_c, y_c)


    def show(self, *args, **kwargs):
        self.closed_loop_system.show(*args, **kwargs)