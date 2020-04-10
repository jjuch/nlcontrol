from sympy.physics.mechanics import msubs
from sympy.matrices import Matrix
from sympy.tensor.array import Array
from sympy import diff, Symbol

from simupy.systems.symbolic import DynamicalSystem

from nlcontrol.systems import SystemBase

from itertools import chain

class EulerLagrange(SystemBase):
    """
    EulerLagrange(states, inputs, sys=None)

    A class that defines SystemBase object using an Euler-Lagrange formulation:
        M(x).x'' + C(x, x').x' + K(x)= F(u)
    Here, x represents a minimal state:
        [x_1, x_2, ...]
    the apostrophe represents a time derivative, and u is the input vector:
        [u_1, u_2, ...]
    A SystemBase object uses a state equation function of the form:
        x' = f(x, u)
    However, as system contains second time derivatives of the state, an extended state x* is necessary containing the minimized states and its first time derivatives:
        x* = [x_1, x_1', x_2, x_2', ...]
    which makes it possible to adhere to the SystemBase formulation:
        x*' = f(x*, u)

    Parameters:
    -----------
        states : string or array-like
            if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols.
        inputs : string or array-like
            if `inputs` is a string, it is a comma-separated listing of the input names. If `inputs` is array-like it contains the inputs as sympy's dynamic symbols.
        sys : simupy's DynamicalSystem object (simupy.systems.symbolic), optional
            the object containing output and state equations, default: None.

    Attributes:
    -----------
        M : inertia_matrix
            Inertia forces (positive definite, symmetric).
        C : damping_matrix
            Damping and Coriolis forces.
        K : elastic_matrix
            Elastic en centrifugal forces.
        F : force_vector
            force or torque vector, non-square matrix.
        states : sympy array of dynamicsymbols
            state variables.
        dstates : sympy array of dynamicsymbols
            first time derivatives of state variables
        inputs : sympy array of dynamicsymbols
            input variables.
        sys : system
            a simupy object 'DynamicalSystem'
                states : the extended states
                inputs : see above
                state_equation : of the extended states (see create_state_equation())
                output_equations: the extended states

    Examples:
    ---------
        * Create a EulerLagrange object with two states and two inputs:
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

        * Get the Euler-Lagrange matrices and the state equations:
            >>> M = sys.inertia_matrix
            >>> C = sys.damping_matrix 
            >>> K = sys.elastic_matrix
            >>> F = sys.force_vector
            >>> xdot = sys.state_equation

        * Linearize an Euler-Lagrange system around the state's working point [0, 0, 0, 0] and the input's working point = [0, 0] and simulate for a step input and initial conditions
            >>> sys_lin, _ = sys.linearize([0, 0, 0, 0], [0, 0])
            >>> from nlcontrol.signals import step
            >>> step_sgnl = step(2)
            >>> init_cond = [1, 2, 0.5, 4]
            >>> sys_lin.simulation(5, initial_conditions=init_cond, input_signals=step_sgnl, plot=True)
    """


    def __init__(self, states, inputs, sys=None):
        minimal_states, extended_states = self.__extend_states__(states)
        self.minimal_states = self.__process_init_input__(minimal_states)
        super().__init__(extended_states, inputs, sys=sys)
        self._M = None
        self._C = None
        self._K = None
        self._F = None
        

    @property
    def inertia_matrix(self) -> Matrix:
        return self._M

    @inertia_matrix.setter
    def inertia_matrix(self, matrix:Matrix):
        if self.check_symmetry(matrix):
            if matrix.shape[0] == len(self.minimal_states):
                self._M = matrix
            else:
                error_text = "[EulerLagrange.inertia_matrix (setter)] The intertia matrix' dimension do not match the minimal state's dimension."
                raise ValueError(error_text) 
        else:
            error_text = '[EulerLagrange.inertia_matrix (setter)] The intertia matrix should be symmetric.'
            raise ValueError(error_text)

    @property
    def damping_matrix(self) -> Matrix:
        return self._C

    @damping_matrix.setter
    def damping_matrix(self, matrix:Matrix):
        if matrix.shape[1] == len(self.minimal_states):
            self._C = matrix
        else:
            error_text = "[EulerLagrange.damping_matrix (setter)] The damping matrix' row length does not match the minimal state's dimension."
            raise ValueError(error_text)

    @property
    def elastic_matrix(self) -> Matrix:
        return self._K

    @elastic_matrix.setter
    def elastic_matrix(self, matrix:Matrix):
        if matrix.shape[0] == len(self.minimal_states):
            self._K = matrix
        else:
            error_text = "[EulerLagrange.elastic_matrix (setter)] The elastic matrix' length does not match the minimal state's dimension."
            raise ValueError(error_text)

    @property
    def force_vector(self) -> Matrix:
        return self._F

    @force_vector.setter
    def force_vector(self, matrix:Matrix):
        if matrix.shape[0] == len(self.minimal_states):
            self._F = matrix
        else:
            error_text = "[EulerLagrange.force_vector (setter)] The force vector's length does not match the minimal state's dimension."
            raise ValueError(error_text)


    def define_system(self, M, C, K, F):
        """
        Define the Euler-Lagrange system using the differential equation representation:
            M(x).x'' + C(x, x').x' + K(x)= F(u)
        Here, x is the minimal state vector created in the constructor. The state-space model is generated in the form x*' = f(x*, u), with x* = [x_1, dx_1, x_2, dx_2, ...], the extended state vector.

        HINT: use create_variables() for an easy notation of state[i] and dstate[i].

        Parameters:
        -----------
            M : array-like
                Inertia matrix, the matrix is positive definite symmetric. Size: n x n
            C : array-like
                Damping matrix. Size: m x n
            K : array-like
                Elastic matrix. Size: n x 1
            F : array-like
                External forces or torque inputs, non-square matrix. Size: n x 1

        Examples:
        ---------
        """
        # Transform to sympy matrices and store to
        M_mat = Matrix(M)
        self.inertia_matrix = M_mat
        C_mat = Matrix(C)
        self.damping_matrix = C_mat
        K_mat = Matrix(K)
        self.elastic_matrix = K_mat
        F_mat = Matrix(F)
        self.force_vector = F_mat

        state_equations = self.create_state_equations()
        self.system = DynamicalSystem(state_equation=state_equations, output_equation=self.states, state=self.states, input_=self.inputs)

    
    def __extend_states__(self, states):
        """
        Create both the minimal and extended state vector in string format. The extended state consists of the minimal states extended with its derivatives:
            minimal states : x_1, x_2, ...
            extended states : x_1, dx_1, x_2, dx_2
        with dx_i representing the derivative of state x_i

        Parameters:
        -----------
            states : str or array-like
                if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols.
        
        Returns:
        --------
            minimal_states: str
                The minimal states of the system as a string.
            exented_states: str or array-like
                if `states`is a string the extended states is returned as a string, else the `states` array-like object is passed on without any adaptations. 
        """
        separator = ", "
        if isinstance(states, str):
            if "," in states:
                extended_state_strings = list(chain.from_iterable( \
                    (state_str.replace(" ", ""),"d" + state_str.replace(" ", "")) \
                    for state_str in states.split(',')))
            else:
                extended_state_strings = [states.replace(" ", ""), \
                    "d" + states.replace(" ", "")]
            minimal_states = states
            return minimal_states, separator.join(extended_state_strings) 
        else:
            extended_states = states
            processed_extended_states =  [str(state).replace("(t)", "") for state in states]
            # keep elements on even index - remove derivative states
            minimal_states = separator.join(processed_extended_states[0::2])
            return minimal_states, extended_states
        

    def create_variables(self, input_diffs:bool=False):
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
        ---------
            * Return the variables of `sys', which has two states and two inputs and add a system to the EulerLagrange object:
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
        
        Parameter:
        ----------
            matrix : sympy matrix
                a matrix that needs to be checked.

        Returns:
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

        Returns:
        --------
            result : sympy array object
                the state equation for each element in self.states
        """
        minimal_dstates = Matrix(self.states[1::2])
        dstates = Matrix(self.dstates[0::2])
        substitution = dict(zip(dstates, minimal_dstates))

        M_inv = self.inertia_matrix.inv()
        states_dotdot = M_inv * self.force_vector \
            - M_inv * self.damping_matrix * minimal_dstates \
            - M_inv * self.elastic_matrix
        states_dot = []
        for i in range(len(self.states)):
            if i%2 == 0:
                states_dot.append(self.states[i+1])
            else:
                states_dot.append( \
                    msubs(states_dotdot[i//2].copy(), substitution))
        states_dot = Array(states_dot)
        return states_dot