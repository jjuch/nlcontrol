from simupy.block_diagram import BlockDiagram, SimulationResult
from simupy.systems.symbolic import DynamicalSystem, MemorylessSystem
from sympy.matrices import Matrix
from sympy.physics.mechanics import msubs
from sympy.tensor.array import Array

from nlcontrol.closedloop.blocks import gain_block
from nlcontrol.systems import SystemBase, ControllerBase
from nlcontrol.visualisation import ClosedLoopRenderer

import numpy as np
import matplotlib.pyplot as plt
import warnings

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

        TODO: not tested yet
        '''
        # Create input vector
        input_dim = self.forward_system.system.dim_input
        inputs = SystemBase.__format_dynamic_vectors__(None, 'r0:{}'.format(input_dim))
        # Obtain new state equations including inputs, TODO
        states, state_equations, output_equations = self.__get_dynamic_equations__(inputs)
    

        # Define a simupy DynamicalSystem
        if states is None and state_equations is None and output_equations is None:
            sys = SystemBase(
                states=None,
                inputs=None,
                name="closed-loop",
                block_type="closedloop",
                forward_sys=self.forward_system,
                backward_sys=self.backward_system
            )
        else:
            sys = SystemBase(
                states=Array(states), 
                inputs=inputs, 
                name="closed-loop", 
                block_type="closedloop", 
                forward_sys=self.forward_system, 
                backward_sys=self.backward_system)
            sys.set_dynamics(output_equations, state_equation=state_equations)
        return sys
        


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



    def __get_dynamic_equations__(self, inputs):
        '''
        Contcatenate the states vector of the system and the controller.

        Parameters
        -----------
        inputs : TODO

        Returns
        --------
        states : list
            first the states of the system and next the states of the controller.
        state_equations : list
            first the state equations of the system and next the state equations of the controller.
        '''
        states = []
        state_equations = []

        fwd_output_equations, bwd_output_equations = self.__get_output_equations__(inputs)
        
        if None in fwd_output_equations or None in bwd_output_equations:
            states = None
            state_equations = None
            fwd_output_equations = None
        else:
            if self._fwd_system is None:
                if self._bwd_system is None:
                    error_text = '[ClosedLoop.__get_states__] Both controller and system are None. One of them should at least contain a system.'
                    raise AssertionError(error_text)
                else:
                    if self._bwd_system.states is not None:
                        states.extend(self._bwd_system.states)

                        unprocessed_substit_bwd_system = zip(self._bwd_system.inputs, input - bwd_output_equations)
                        minimal_dstates = self._bwd_system.states[1::2]
                        dstates = self._bwd_system.dstates[0::2]
                        substitutions_derivatives = dict(zip(dstates, minimal_dstates))
                        substitutions_system = dict([(k, msubs(v, substitutions_derivatives))\
                        for k, v in unprocessed_substit_bwd_system])

                        bwd_state_eq = [msubs(state_eq, substitutions_derivatives, substitutions_system)\
                        for state_eq in self._bwd_system.state_equation]

                        state_equations.extend(bwd_state_eq)
            else:
                substitutions_derivatives = dict()
                if self._bwd_system is None:
                    #unprocessed_substitutions_system = zip(self._fwd_system.inputs, (-1) * self._fwd_system.output_equation)
                    unprocessed_substitutions_system = zip(self._fwd_system.inputs, inputs - fwd_output_equations)
                else:
                    #unprocessed_substitutions_system = zip(self._fwd_system.inputs, (-1) * self._bwd_system.output_equation)
                    unprocessed_substitutions_system = zip(self._fwd_system.inputs, inputs - bwd_output_equations)
                state_eq_temp = []
                if self._fwd_system.states is not None:
                    # Remove Derivative(., 't') from controller states and substitutions_system
                    n_fwd_states = len(self._fwd_system.states)
                    n_2_fwd_states = int(np.floor(n_fwd_states/2)) 
                    minimal_dstates = self._fwd_system.states[n_2_fwd_states:None]
                    dstates = self._fwd_system.dstates[0:n_2_fwd_states]
                    substitutions_derivatives = dict(zip(dstates, minimal_dstates))
                    substitutions_system = dict([(k, msubs(v, substitutions_derivatives))\
                        for k, v in unprocessed_substitutions_system])

                    states.extend(self._fwd_system.states)
                    state_eq_temp = [msubs(state_eq, substitutions_derivatives, substitutions_system)\
                        for state_eq in self._fwd_system.state_equation]            
                if self._bwd_system is not None and self._bwd_system.states is not None:
                    state_equations.extend(state_eq_temp)
                    states.extend(self._bwd_system.states)
                    unprocessed_substit_bwd_system = zip(self._bwd_system.inputs, fwd_output_equations)

                    n_bwd_states = len(self._bwd_system.states)
                    n_2_bwd_states = int(np.floor(n_bwd_states/2)) 
                    minimal_dstates = self._bwd_system.states[n_2_bwd_states:None]
                    dstates = self._bwd_system.dstates[0:n_2_bwd_states]

                    substitutions_derivatives = dict(zip(dstates, minimal_dstates))
                    substitutions_system = dict([(k, msubs(v, substitutions_derivatives))\
                        for k, v in unprocessed_substit_bwd_system])
                    
                    controller_state_eq = [msubs(state_eq, substitutions_derivatives, substitutions_system)\
                        for state_eq in self._bwd_system.state_equation]

                    state_equations.extend(controller_state_eq)   
                else:
                    bwd_inputs = self._bwd_system.inputs
                    substitutions_bwd_inputs = dict(zip(bwd_inputs,fwd_output_equations))
                    state_equations.extend([msubs(state_eq, substitutions_bwd_inputs)\
                        for state_eq in state_eq_temp])
        return states, state_equations, fwd_output_equations
    
    def __get_output_equations__(self, inputs):
        fwd_output_equations = []
        bwd_output_equations = []
        if self._fwd_system is None:
            if self._bwd_system is None:
                error_text = '[ClosedLoop.__get_output__] Both controller and system are None. One of them should at least contain a system.'
                raise AssertionError(error_text)
            else:
                unprocessed_substitutions_system = zip(self._bwd_system.inputs, inputs - self._bwd_system.output_equation)
                minimal_dstates = self._bwd_system.states[1::2]
                dstates = self._bwd_system.dstates[0::2]
                substitutions_derivatives = dict(zip(dstates, minimal_dstates))
                substitutions_system = dict([(k, msubs(v, substitutions_derivatives))\
                    for k, v in unprocessed_substitutions_system])

                bwd_output_equations.extend([msubs(output_eq, substitutions_derivatives, substitutions_system)\
                    for output_eq in self._bwd_system.output_equation])
        
        else:
            if self._bwd_system is None:
                if self._fwd_system._additive_output_system is not None:
                    warning_text = "[ClosedLoop.__get_output__] Unable to create closed-form output equations when the output equation of the forward system contains an input. It is possible to simulate this case with the 'simulation' function of the ClosedLoop object."
                    # raise AssertionError(error_text)
                    warnings.warn(warning_text)
                    fwd_output_equations.append(None)
                    bwd_output_equations.append(None)
                else:
                    unprocessed_fwd_substitution_system = zip(self._fwd_system.inputs, inputs - self._fwd_system.output_equation)
                    minimal_dstates = self._fwd_system.states[1::2]
                    dstates = self._fwd_system.dstates[0::2]
                    substitutions_derivatives = dict(zip(dstates, minimal_dstates))
                    substitutions_system = dict([(k, msubs(v, substitutions_derivatives))\
                        for k, v in unprocessed_fwd_substitution_system])
                
                fwd_output_equations.extend([msubs(output_eq, substitutions_derivatives, substitutions_system)\
                    for output_eq in self._fwd_system.output_equation])
            else:
                if self._bwd_system._additive_output_system is not None:
                    warning_text = "[ClosedLoop.__get_output__] Unable to create closed-form output equations when the output equation of the backward system contains an input. It is possible to simulate this case with the 'simulation' function of the ClosedLoop object."
                    warnings.warn(warning_text)
                    fwd_output_equations.append(None)
                    bwd_output_equations.append(None)
                else:
                    unprocessed_fwd_substitution_system = zip(self._fwd_system.inputs, inputs - self._bwd_system.output_equation)
                    minimal_dstates = self._fwd_system.states[1::2]
                    dstates = self._fwd_system.dstates[0::2]
                    substitutions_derivatives = dict(zip(dstates, minimal_dstates))
                    substitutions_system = dict([(k, msubs(v, substitutions_derivatives))\
                        for k, v in unprocessed_fwd_substitution_system])
                
                    fwd_output_equations.extend([msubs(output_eq, substitutions_derivatives, substitutions_system)\
                        for output_eq in self._fwd_system.output_equation])
                    bwd_output_equations.extend(self._bwd_system.output_equation)
        return Array(fwd_output_equations), Array(bwd_output_equations)


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

    
    def add_system(self, system: SystemBase or MemorylessSystem, BD=None):
        """
        Add a system to SimuPy's `BlockDiagram`. It takes into account that the new system can contain an additive system. This will result in the addition of a summation on the output, the base system and the additive system.

        Parameters
        -----------
        system : SystemBase or SimuPy.MemorylessSystem
            The system that needs to be added to a SimuPy.BlockDiagram object.
        BD : SimuPy.BlockDiagram, optional
            The block-diagram to which the system needs to be added. If none is given a new one is made.

        Returns
        --------
        BD [SimuPy.BlockDiagram] : the block-diagram containing the system.
        total_output_size [int] : the number of output variables that are added to the BD. With additive systems this will be the output dimension of the summation and the additive system as well.
        """
        if not isinstance(system, (SystemBase, MemorylessSystem)):
            error_text = "[ClosedLoop.add_system] The system to be added should be of the type SystemBase or any derived classes or a `SimuPy`'s MemorylessSystem."
            raise TypeError(error_text)
        if BD is None:
            BD = BlockDiagram()
        elif not isinstance(BD, BlockDiagram):
            error_text = "[ClosedLoop.add_system] The BD should be None or of the type Blockdiagram."
            raise TypeError(error_text)
        total_output_size = 0
        if (not isinstance(system, MemorylessSystem)) and (system._additive_output_system is not None):
            output_system = system._additive_output_system
            base_system = system.system
            output_dim = output_system.dim_output
            if output_dim != base_system.dim_output:
                error_text = "[ClosedLoop.add_system] The dimension of the output_system and base_system should be the same."
                raise AssertionError(error_text)
            import nlcontrol.closedloop.blocks as blocks
            summation = blocks.summation_block(output_dim)
            BD.add_system(summation)
            total_output_size += summation.dim_output
            BD.add_system(output_system)
            total_output_size += output_system.dim_output
            BD.add_system(base_system)
            total_output_size += base_system.dim_output
            # Connections to the summation block
            indices_base_system = [i for i in range(output_dim)]
            indices_output_system = [output_dim + el for el in indices_base_system]
            BD.connect(base_system, summation, inputs=indices_base_system)
            BD.connect(output_system, summation, inputs=indices_output_system)
        elif isinstance(system, MemorylessSystem):
            BD.add_system(system)
            total_output_size += system.dim_output
        else:
            BD.add_system(system.system)
            total_output_size += system.system.dim_output
        return BD, total_output_size


    def connect(self, system1: SystemBase or MemorylessSystem, system2: SystemBase or MemorylessSystem, BD: BlockDiagram):
        """ 
            Connects two systems contained in a SimuPy's `BlockDiagram`. It takes into account the additive system in the `SystemBase` objects.

            Parameters
            -----------
            system1 : SystemBase or SimuPy.MemorylessSystem
                The system that needs to be connected to system2.
            system2 : SystemBase or SimuPy.MemorylessSystem
                The system to which system1 needs to be connected.
            BD : SimuPy.BlockDiagram
                The block-diagram which contains system1 and system2.

            Returns
            --------
            BD [SimuPy.BlockDiagram] : the block-diagram with the new connection.
        """
        if not isinstance(BD, BlockDiagram):
            error_text = "[ClosedLoop.connect] The BD should be None or of the type Blockdiagram."
            raise TypeError(error_text)
        if (not isinstance(system1, (SystemBase, MemorylessSystem))) or (not isinstance(system2, (SystemBase, MemorylessSystem))):
            error_text = "[ClosedLoop.connect] the systems that need to be connected should be of the type `SystemBase` or SimuPy's `MemorylessSystem`."
            raise TypeError(error_text)
        systems_in_BD = list(BD.systems)
        idx_input2 = []
        idx_output1 = None
        # Find indices of the input subsystems in BD of the systems that need to be connected.
        if (not isinstance(system2, MemorylessSystem)) and system2._additive_output_system is not None:
            try:
                idx_input2.append(systems_in_BD.index(system2.system)) # index of main system
                idx_input2.append(systems_in_BD.index(system2._additive_output_system)) # index of additive system
            except ValueError:
                error_text = "[ClosedLoop.connect] The system is not properly added to the BD. Make sure that you use `ClosedLoop`'s add_system method first."
                raise ValueError(error_text)
        else:
            try:
                if isinstance(system2, MemorylessSystem):
                    sys2 = system2
                else:
                    sys2 = system2.system
                idx_input2.append(systems_in_BD.index(sys2)) # index of main system
            except ValueError:
                error_text = "[ClosedLoop.connect] The system is not properly added to the BD. Make sure that you use `ClosedLoop`'s add_system method first."
                raise ValueError(error_text)
        if (not isinstance(system1, MemorylessSystem)) and system1._additive_output_system is not None:
            try:
                idx_output1 = systems_in_BD.index(system1._additive_output_system) - 1 # The summation is added right before the additive system
            except ValueError:
                error_text = "[ClosedLoop.connect] The system is not properly added to the BD. Make sure that you use `ClosedLoop`'s add_system method first."
                raise ValueError(error_text)
        else:
            try:
                if isinstance(system1, MemorylessSystem):
                    sys1 = system1
                else:
                    sys1 = system1.system
                idx_output1 = systems_in_BD.index(sys1)
            except ValueError:
                error_text = "[ClosedLoop.connect] The system is not properly added to the BD. Make sure that you use `ClosedLoop`'s add_system method first."
                raise ValueError(error_text)
        for input_idx in idx_input2:
            BD.connect(systems_in_BD[idx_output1], systems_in_BD[input_idx])
        return BD


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
        # Order of adding systems is important. The trick is to cut the loop where the feedback ends and work from that point back to the beginning. This can be seen from simupy.block_diagram.BlockDiagram.output_equation_function(). Possibly only for stateless systems important.

        # Keep slices per block to find the proper indices.
        states_forward_slices = []
        states_backward_slices = []
        output_forward_slices = []
        output_backward_slices = []

        current_cumm_state = 0
        current_cumm_output = 0

        # Cut loop at on the feedback node and add systems from there backwards to the block diagram
        if (backward_systems[0] is not None):
            negative_feedback = gain_block(-1, backward_systems[-1].system.dim_output)
            BD, _ = self.add_system(negative_feedback, BD=BD)
            current_cumm_output += negative_feedback.dim_output
            
            # Add each backward system to BD
            backward_systems.reverse() # Reverse the list
            for i, backward_system in enumerate(backward_systems):
                BD, system_output_size = self.add_system(backward_system, BD=BD)
                # Update index slices #TODO : slices for systems with additive systems are not correct
                system_state_slice = slice(current_cumm_state, current_cumm_state + backward_system.system.dim_state)
                system_output_slice = slice(current_cumm_output, current_cumm_output + backward_system.system.dim_output)
                states_backward_slices.append(system_state_slice)
                output_backward_slices.append(system_output_slice)
                current_cumm_state = current_cumm_state + backward_system.system.dim_state
                current_cumm_output = current_cumm_output + system_output_size

                # Connect systems
                if i == 0:
                    BD = self.connect(backward_system, negative_feedback, BD)
                else:
                    BD = self.connect(backward_system, backward_systems[i - 1], BD)
        else:
            # No blocks in the backward path
            negative_feedback = gain_block(-1, forward_systems[-1].system.dim_output)
            BD, __import__ = self.add_system(negative_feedback, BD=BD)
            current_cumm_output += negative_feedback.dim_output

        # Add each backward system to BD
        forward_systems.reverse()
        if (forward_systems[0] is not None):
            for forward_system in forward_systems:
                BD, system_output_size = self.add_system(forward_system, BD=BD)
                # Update index slices
                system_state_slice = slice(current_cumm_state, current_cumm_state + forward_system.system.dim_state)
                system_output_slice = slice(current_cumm_output, current_cumm_output + forward_system.system.dim_output)
                states_forward_slices.append(system_state_slice)
                output_forward_slices.append(system_output_slice)
                current_cumm_state = current_cumm_state + forward_system.system.dim_state
                current_cumm_output = current_cumm_output + system_output_size

        for i in range(len(forward_systems)):
            if (i == 0): #Last forward system as the forward systems are reversed
                if backward_systems[0] is not None:
                    BD = self.connect(forward_systems[i], backward_systems[-1], BD) # backward systems are also reversed
                else:
                    BD = self.connect(forward_systems[i], negative_feedback, BD)
            else:
                BD = self.connect(forward_systems[i], forward_systems[i - 1], BD)

        # connect the negative gain to first forward block
        BD = self.connect(negative_feedback, forward_systems[-1], BD)
        
        # Pack indices
        indices = {
            'process': {
                'output': output_forward_slices,
                'state': states_forward_slices
            },
            'controller': {
                'output': output_backward_slices,
                'state': states_backward_slices
            }
        }
        return BD, indices


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
            if len(system_with_states) != 0:
                cum_states = 0
                for sys in system_with_states:
                    sys.initial_condition = res.x[index][cum_states:cum_states + sys.dim_state]
                    cum_states = cum_states + sys.dim_state
        return res


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
        if np.isscalar(tspan):
            res = self.block_diagram.simulate(tspan, integrator_options=integrator_options)
        elif len(tspan) == 2:
            res = self.block_diagram.simulate(tspan, integrator_options=integrator_options)
        else:
            system_with_states = []
            for sys in self.block_diagram.systems:
                if hasattr(sys, 'dim_state') and sys.dim_state > 0:
                    system_with_states.append(sys)
            res = self.__simulation_loop__(tspan, self.block_diagram, system_with_states, integrator_options)
        
        # Unpack indices
        y_p_idx = self.indices['process']['output']
        x_p_idx = self.indices['process']['state']
        y_c_idx = self.indices['controller']['output']
        x_c_idx = self.indices['controller']['state']

        # slice results into correct vectors
        y_p = self.__slice_simulation_results__(y_p_idx, res.y)
        x_p = self.__slice_simulation_results__(x_p_idx, res.x)
        y_c = self.__slice_simulation_results__(y_c_idx, res.y)
        x_c = self.__slice_simulation_results__(x_c_idx, res.x)

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
    

    def __slice_simulation_results__(self, slice_indices: list, data, invert=True):
        """
        A helper function to split a multi-dimensional data set with a list of slices. The invert boolean indicates whether each slice is prepended or appended to the resulting data matrix.

        Parameters
        -----------
        slice_indices : list
            a list containing slices.
        data : ndarray
            a data set which needs to be sliced.
        invert : boolean
            indicates whether each new slice needs to prepended or appended to the resulting matrix.

        Returns
        --------
        ndarray or None : the sliced dataset.        
        """
        result = None
        if len(slice_indices) != 0 :
            for sl in slice_indices:
                if sl.start != sl.stop:
                    data_slice = data[:, sl]
                    if result is None:
                        # initialize result before using concatenate function
                        result = data_slice
                    else:
                        if invert:
                            result = np.concatenate((data_slice, result), axis=1)
                        else:
                            result = np.concatenate((result, data_slice), axis=1)
        return result


    def show(self, *args, **kwargs):
        self.closed_loop_system.show(*args, **kwargs)