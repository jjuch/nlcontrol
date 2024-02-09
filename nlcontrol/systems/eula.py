from sympy.physics.mechanics import msubs
from sympy.matrices import Matrix, zeros
from sympy.tensor.array import Array, NDimArray
from sympy import diff, Symbol, Expr

from simupy.systems.symbolic import DynamicalSystem

from nlcontrol.systems import SystemBase

import math
import numpy as np

__all__ = ["EulerLagrange"]

class EulerLagrange(SystemBase):
    """
    EulerLagrange(states, inputs, diff_inputs=False, system=None, name="EL system")

    A class that defines SystemBase object using an Euler-Lagrange formulation:

    .. math::
        M(x).x'' + C(x, x').x' + K(x) + Qrnc(x') = F(u, u')

    Here, x represents a minimal state:

    .. math::
        x = [x_1, x_2, ...]^T

    the apostrophe represents a time derivative, :math:`.^T` is the transpose of a matrix, and u is the input vector:

    .. math::
        u = [u_1, u_2, ...]^T
    
    If diff_inputs is True the input vector is

    .. math::
        u^{*} = [u_1, u_2, ..., du_1, du_2, ...]
    
    If diff_inputs is False the input vector is

    .. math::
        u^{*} = u

    A SystemBase object uses a state equation function of the form:

    .. math::
        x' = h(x, u^{*})

    However, as system contains second time derivatives of the state, an extended state x* is necessary containing the minimized states and its first time derivatives:

    .. math::
        x^{*} = [x_1, x_2, x_1', x_2', ...]

    which makes it possible to adhere to the SystemBase formulation:

    .. math::
        x^{*'} = f(x^{*}, u^{*})

    The output is by default the state vector
    
    .. math::
        y = x^{*}

    A custom output equation can be chosen:

    ..math::
        y = g(x^{*}, u^{*})

    Parameters
    -----------
    states : string or array-like
        if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols.
    inputs : string or array-like
        if `inputs` is a string, it is a comma-separated listing of the input names. If `inputs` is array-like it contains the inputs as sympy's dynamic symbols.
    diff_inputs : boolean, optional
        if true the input vector is expanded with the input derivatives. This also means that the input's dimension is doubled.
    system : simupy's DynamicalSystem object (simupy.systems.symbolic), optional
        the object containing output and state equations, default: None.
    name : string
        give the system a custom name which will be shown in the block scheme, default: 'EL system'.


    Examples
    ---------
    * Create a EulerLagrange object with two states and two inputs:
        >>> from sympy import atan
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
        >>> Qrnc = [atan(dx1), 0]
        >>> g = [x1, x1, x2, x2]
        >>> sys.define_system(M, C, K, F, Qrnc=Qrnc, g=g)

    * Get the Euler-Lagrange matrices and the state equations:
        >>> M = sys.inertia_matrix
        >>> C = sys.damping_matrix 
        >>> K = sys.stiffness_matrix
        >>> F = sys.input_vector
        >>> xdot = sys.state_equation

    * Linearize an Euler-Lagrange system around the state's working point [0, 0, 0, 0] and the input's working point = [0, 0] and simulate for a step input and initial conditions
        >>> sys_lin, _ = sys.linearize([0, 0, 0, 0], [0, 0])
        >>> from nlcontrol.signals import step
        >>> step_sgnl = step(2)
        >>> init_cond = [1, 2, 0.5, 4]
        >>> sys_lin.simulation(5, initial_conditions=init_cond, input_signals=step_sgnl, plot=True)
    """

    def __init__(self, states, inputs, **kwargs):
        minimal_states, extended_states = self.__extend_states__(states)
        self.minimal_states = self.__format_dynamic_vectors__(minimal_states)
        if 'system' not in kwargs:
            kwargs['system'] = None
        if 'name' not in kwargs:
            kwargs['name'] = "EL system"
        if 'diff_inputs' not in kwargs:
            self._dinputs_bool = False
            dinputs_memory_bool = self._dinputs_bool
        else:
            self._dinputs_bool = kwargs['diff_inputs']
            del kwargs['diff_inputs']
            # Due to running the self.inputs setter twice (see line below else and super().__init__) the self._dinputs_bool needs to be reset in between the two calls. To avoid losing this information the bool is set to it correct value again afterwards. This is a dirty solution and should be solved more elegantly if possible.
            dinputs_memory_bool = self._dinputs_bool
        self._dinputs_bool = dinputs_memory_bool
        self._M = None
        self._C = None
        self._K = None
        self._Qrnc = None
        self._F = None
        self._y = None

        self.inputs = inputs
        super().__init__(extended_states, self.inputs, block_type='system', **kwargs)
    
    
    def __str__(self):
        return """
        EulerLagrange object:
        =====================
        M(x).x'' + C(x, x').x' + K(x) + Qrnc(x') = F(u, u')\n
        \twith:
        \t\tx: {}
        \t\tM: {}
        \t\tC: {}
        \t\tK: {}
        \t\tQrnc: {}
        \t\tF: {}

        \t\toutput: {}
        """.format(self.minimal_states, \
                self.__matrix_to_str__(self.inertia_matrix), \
                self.__matrix_to_str__(self.damping_matrix), \
                self.__matrix_to_str__(self.stiffness_matrix), \
                self.__matrix_to_str__(self.non_conservative_force), \
                self.__matrix_to_str__(self.input_vector),\
                self.__matrix_to_str__(self.output_equation))
    
    def __matrix_to_str__(self, matrix):
        """
        Convert sympy's matrix object 
            Matrix([[a, Derivative(b, t)], [c, d]])
        to the string
            '''
            [[a,            Derivative(b, t)],
            [c          d]]
            '''
        which can be used in the EulerLagrange __str__ method.
        """
        return str(matrix).replace("Matrix(", "").replace("]])", "]]").replace("],", "],\n\t\t\t\t").replace(", ", ", \t\t\t").replace(", \t\t\tt)", ", t)")

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, new_inputs):
        inputs = self.__format_dynamic_vectors__(new_inputs)
        # If the derivative inputs need to be included
        if self._dinputs_bool:
            dinputs = self.__format_dynamic_vectors__(new_inputs, 1)
            inputs = Array(inputs.tolist() + dinputs.tolist())
            # The inputs setter is called twice (in init and in super.init), to avoid adding second derivatives the bool needs to be reset temporarily. The bool is restored in the init. This is a dirty solution and needs to be fixed in the future.
            self._dinputs_bool = False
        self._inputs = inputs

    @property
    def inertia_matrix(self) -> Matrix:
        """
        :obj:`sympy Matrix`

        The matrix represents the inertia forces and it is checked that it is positive definite and symmetric. More on `sympy's Matrix <https://docs.sympy.org/latest/modules/matrices/dense.html#matrix-class-reference>`__.
        """
        return self._M

    @inertia_matrix.setter
    def inertia_matrix(self, matrix:tuple or Matrix):
        if len(matrix) == 2:
            update_system = matrix[1]
            matrix = matrix[0]
        else:
            update_system = True
        if self.check_symmetry(matrix):
            if matrix.shape[0] == len(self.minimal_states):
                self._M = matrix
                if update_system:
                    self.define_system()
            else:
                error_text = "[EulerLagrange.inertia_matrix (setter)] The intertia matrix' dimension do not match the minimal state's dimension."
                raise ValueError(error_text)
        else:
            error_text = '[EulerLagrange.inertia_matrix (setter)] The intertia matrix should be symmetric.'
            raise ValueError(error_text)

    @property
    def damping_matrix(self) -> Matrix:
        """
        :obj:`sympy Matrix`

        The matrix represents the damping and coriolis forces. More on `sympy's Matrix <https://docs.sympy.org/latest/modules/matrices/dense.html#matrix-class-reference>`__.
        """
        return self._C

    @damping_matrix.setter
    def damping_matrix(self, matrix:tuple or Matrix):
        if len(matrix) == 2:
            update_system = matrix[1]
            matrix = matrix[0]
        else:
            update_system = True
        if matrix.shape[1] == len(self.minimal_states):
            self._C = matrix
            if update_system:
                self.define_system()
        else:
            error_text = "[EulerLagrange.damping_matrix (setter)] The damping matrix' row length does not match the minimal state's dimension."
            raise ValueError(error_text)

    @property
    def stiffness_matrix(self) -> Matrix:
        """
        :obj:`sympy Matrix`

        The matrix represents the elastic and centrifugal forces. More on `sympy's Matrix <https://docs.sympy.org/latest/modules/matrices/dense.html#matrix-class-reference>`__.
        """
        return self._K

    @stiffness_matrix.setter
    def stiffness_matrix(self, matrix:tuple or Matrix):
        if len(matrix) == 2:
            update_system = matrix[1]
            matrix = matrix[0]
        else:
            update_system = True
        if matrix.shape[0] == len(self.minimal_states):
            self._K = matrix
            if update_system:
                self.define_system()
        else:
            error_text = "[EulerLagrange.stiffness_matrix (setter)] The elastic matrix' length does not match the minimal state's dimension."
            raise ValueError(error_text)

    @property
    def non_conservative_force(self) -> Matrix:
        """
        :obj:`sympy Matrix`

        The matrix represents the external force or torque vector. This is a non-square matrix. More on `sympy's Matrix <https://docs.sympy.org/latest/modules/matrices/dense.html#matrix-class-reference>`__.
        """
        return self._Qrnc

    @non_conservative_force.setter
    def non_conservative_force(self, matrix:tuple or Matrix):
        if len(matrix) == 2:
            update_system = matrix[1]
            matrix = matrix[0]
        else:
            update_system = True
        if matrix.shape[0] == len(self.minimal_states):
            self._Qrnc = matrix
            if update_system:
                self.define_system()
        else:
            error_text = "[EulerLagrange.non_conservative_force (setter)] The non_conservative force vector's length does not match the minimal state's dimension."
            raise ValueError(error_text)

    @property
    def input_vector(self) -> Matrix:
        """
        :obj:`sympy Matrix`

        The matrix represents the external force or torque vector. This is a non-square matrix. More on `sympy's Matrix <https://docs.sympy.org/latest/modules/matrices/dense.html#matrix-class-reference>`__.
        """
        return self._F

    @input_vector.setter
    def input_vector(self, matrix:tuple or Matrix):
        if len(matrix) == 2:
            update_system = matrix[1]
            matrix = matrix[0]
        else:
            update_system = True
        if matrix.shape[0] == len(self.minimal_states):
            self._F = matrix
            if update_system:
                self.define_system()
        else:
            error_text = "[EulerLagrange.input_vector (setter)] The force vector's length does not match the minimal state's dimension."
            raise ValueError(error_text)
        
    @property
    def output_vector(self) -> Matrix:
        """
        :obj:`sympy Matrix`

        The matrix represents the output vector. This is a non-square matrix. More on `sympy's Matrix <https://docs.sympy.org/latest/modules/matrices/dense.html#matrix-class-reference>`__.
        """
        return self._y
    
    @output_vector.setter
    def output_vector(self, matrix:tuple or Matrix):
        if len(matrix) == 2:
            update_system = matrix[1]
            matrix = matrix[0]
        else:
            update_system = True
        self._y = matrix
        if update_system:
            self.define_system()


    def define_system(self, M=None, C=None, K=None, F=None, Qrnc=None, g=None):
        """
        Define the Euler-Lagrange system using the differential equation representation:

        .. math::
            M(x).x'' + C(x, x').x' + K(x) + Qrnc(x') = F(u, u')
        
        Here, x is the minimal state vector created in the constructor. The state-space model is generated in the form :math:`x^{*'} = f(x^*, u, u')`, with :math:`x^* = [x_1, x_2,..., dx_1, dx_2, ...]`, the extended state vector. The input vector is u^{*} = [u1, u2,..., du1, du2,...]
        The output equation is by default given by

        .. math::
            y = x^{*}

        or a custom output equation
        
        .. math::
            y = g(x^{*}, u)

        .. note:: Use create_variables() for an easy notation of state[i] and dstate[i]. If no matrix is specified for M, C, K, and F, the class object's matrix is used.

        Parameters:
        -----------
        M : array-like or float, optional (default: None)
            Inertia matrix, the matrix is positive definite symmetric. If 'None' is provided, self.inertia_matrix is used. Size: n x n. 
        C : array-like or float, optional (default: None)
            Coriolis matrix. If 'None' is provided, self.damping_matrix is used. Size: m x n
        K : array-like or float, optional (default: None)
            Stiffness matrix. If 'None' is provided, self.stiffness_matrix is used. Size: n x 1
        F : array-like, optional (default: None)
            Input forces/torque, non-square matrix. If 'None' is provided, self.input_vector is used. Size: n x 1
        Qrnc : array-like, optional
            Real non-conservative forces, non-square matrix. Size: n x 1
        g : array-like, optional
            Output equation. Size: p x 1

        Examples:
        ---------
            See class object.
        """

        def condition_input(x):
            if isinstance(x, (float, int, Expr)):
                x_cond = Matrix([[x]])
            elif isinstance(x, (list, np.ndarray, np.matrix)):
                x_cond = Matrix(x)
            else:
                x_cond = x
            return x_cond
        
        # Transform to sympy matrices and store
        if M is not None:
            M_mat = condition_input(M)
            self.inertia_matrix = (M_mat, False)
        if C is not None:
            C_mat = condition_input(C)
            self.damping_matrix = (C_mat, False)
        if K is not None:
            K_mat = condition_input(K)
            self.stiffness_matrix = (K_mat, False)
        if F is not None:
            F_mat = condition_input(F)
            self.input_vector = (F_mat, False)

  
        if Qrnc is not None:
            Qrnc_mat = condition_input(Qrnc)
            self.non_conservative_force = (Qrnc_mat, False)
        elif self.non_conservative_force is None:
            n = len(self.minimal_states)
            Qrnc = zeros(n, 1)
            Qrnc_mat = condition_input(Qrnc)
            self.non_conservative_force = (Qrnc_mat, False)

        if g is not None:
            output_conditioned = condition_input(g)
            self.output_vector = (self.create_output_equation(output_conditioned), False)
        elif self.output_vector is None:
            self.output_vector = (self.states, False)
            
        
        state_equations = self.create_state_equations()
        # Deprecated:
        # self.system = DynamicalSystem(state_equation=state_equations, output_equation=output_equation, state=self.states, input_=self.inputs)
        self.set_dynamics(self.output_vector, state_equation=state_equations)

    
    def __extend_states__(self, states):
        """
        Create both the minimal and extended state vector in string format. The extended state consists of the minimal states extended with its derivatives:
            minimal states : x_1, x_2, ...
            extended states : x_1, x_2, ..., dx_1, dx_2
        with dx_i representing the derivative of state x_i

        Parameters
        -----------
        states : str or array-like
            if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols.
        
        Returns
        --------
        minimal_states: str
            The minimal states of the system as a string.
        exented_states: str or array-like
            if `states`is a string the extended states is returned as a string, else the `states` array-like object is passed on without any adaptations. 
        """
        separator = ", "
        if isinstance(states, str):
            if "," in states:
                states_lst = [state_str.replace(" ", "") for state_str in states.split(',')]
                dstates_lst = ["d" + state_str.replace(" ", "") for state_str in states.split(',')]
                extended_state_strings = states_lst + dstates_lst
            else:
                extended_state_strings = [states.replace(" ", ""), \
                    "d" + states.replace(" ", "")]
            minimal_states = states
            return minimal_states, separator.join(extended_state_strings) 
        else:
            extended_states = states
            processed_extended_states =  [str(state).replace("(t)", "") for state in states]
            # keep first half of state vector
            len_extended_states = len(states)
            minimal_states = separator.join(processed_extended_states[0:math.floor(len_extended_states/2)])
            return minimal_states, extended_states
        

    def create_variables(self, input_diffs:bool=False):
        """
        Returns a tuple with all variables. First the states are given, next the derivative of the states, and finally the inputs, optionally followed by the diffs of the inputs. All variables are sympy dynamic symbols.

        Parameters
        -----------
        input_diffs : boolean
            also return the differentiated versions of the inputs, default: false.

        Returns
        --------
        variables : tuple
            all variables of the system.

        Examples
        ---------
        * Return the variables of 'sys', which has two states and two inputs and add a system to the EulerLagrange object:
            >>> x1, x2, x1dot, x2dot, u1, u2, u1dot, u2dot = sys.create_variables(input_diffs=True)
            >>> M = [[1, x1*x2],
                [x1*x2, 1]]
            >>> C = [[2*x1dot, 1 + x1],
                [x2 - 2, 3*x2dot]]
            >>> K = [x1, 2*x2]
            >>> F = [u1, 0]
            >>> sys.define_system(M, C, K, F)
        """
        return super().create_variables(states=self.minimal_states)


    def check_symmetry(self, matrix) -> bool:
        """Check if matrix is symmetric. Returns a bool.
        
        Parameter
        ----------
        matrix : sympy matrix
            a matrix that needs to be checked.

        Returns
        --------
        value : bool
            the matrix being symmetric or not.
        """
        matrix_shape = matrix.shape
        el_sym = []
        if matrix_shape[0] == matrix_shape[1]:
            for i in range(matrix_shape[0]):
                for j in range(matrix_shape[1]):
                    el_sym.append(matrix[i,j] == matrix[j, i])
            return all(el_sym)
        else:
            print('Error: matrix is not squared.')
            return False


    def create_state_equations(self):
        """
        As the system contains a second derivative of the states, an extended state should be used, which contains the first derivative of the states as well. Therefore, the state equation has to be adapted to this new state vector.

        Returns
        --------
        result : Sympy array object
            the state equation for each element in self.states
        """
        # Convert self.states to list before slicing, to be compatible with sympy > 1.4
        len_states = len(self.states)
        minimal_dstates = Matrix(self.states.tolist()[math.floor(len_states/2):])
        dstates = Matrix(self.dstates.tolist()[0:math.floor(len_states/2)])
        substitution = dict(zip(dstates, minimal_dstates))
        deriv_subs = self.__create_derivative_substitutions__()
        
        M_inv = self.inertia_matrix._rep.to_field().inv().to_Matrix()
        states_dotdot = M_inv * self.input_vector \
            - M_inv * self.damping_matrix * minimal_dstates \
            - M_inv * self.stiffness_matrix \
            - M_inv * self.non_conservative_force
        # Add velocities to new state equation
        states_dot = [mdst[0] for mdst in minimal_dstates.tolist()]
        # Add accelerations to new state equation
        for i in range(len(states_dotdot)):
            states_dot.append(\
                    msubs(states_dotdot[i].copy(), deriv_subs))

        states_dot = Array(states_dot)
        return states_dot


    def create_output_equation(self, output_eq:Matrix):
        """
        Condition the output equation such that it does not contain Derivative(x(t),t) terms.

        Parameters
        -----------
        output_eq : Sympy matrix object
            The output equation that needs to be conditioned.

        Returns
        --------
        output_equation : Sympy matrix object
            The conditioned output equation.
        """
        deriv_subs = self.__create_derivative_substitutions__()
        output_equation = msubs(output_eq.copy(), deriv_subs)
        return output_equation


    def __create_derivative_substitutions__(self):
        """
        Create a dict for state substitutions of Derivative(x(t),t) into dx(t).

        Returns
        --------
        substitution : dict
            A dict with the substitutions of the derivative terms.
        """
        # Convert self.states to list before slicing, to be compatible with sympy > 1.4
        len_states = len(self.states)
        dstates = Matrix(self.dstates.tolist()[0:math.floor(len_states/2)])
        minimal_dstates = Matrix(self.states.tolist()[math.floor(len_states/2):])
        substitution = dict(zip(dstates, minimal_dstates))
        return substitution