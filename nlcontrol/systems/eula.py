from sympy.physics.mechanics import dynamicsymbols, msubs
from sympy import symbols, cos, sin
from sympy.matrices import Matrix, ones
from sympy.tensor.array import Array
from sympy import diff, Symbol

from simupy.systems.symbolic import DynamicalSystem

from nlcontrol.systems import SystemBase

from itertools import chain

class EulerLagrange(SystemBase):
    """
    A class that defines an Euler-Lagrange formulated Systems.

    Attributes
    ----------
    M: Inertia matrix - Inertia forces (positive definite, symmetric).
    C: Damping matrix - Damping and Coriolis forces.
    K: Elastic matrix - Elastic en centrifugal forces.
    F: External forces, non-square matrix.
    states: String of the position state variables.
    inputs: String of the input variables.
    xdot: State_space representation
    sys: simupy object 'DynamicalSystem'
    """


    def __init__(self, states, inputs, sys=None):
        """
        The underactuated mechanical systems class uses four matrices to describe the equations of motion. An underactuated system is described with the following differential equation: 
            M(q).q'' + C(q, q').q' + K(q)= F(u)

        Parameters:
            states [str]: Position state variables. The variables are separated by ','.
            inputs [str]: input variables. The variables are separated by ','.
        """
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
            M(q).q'' + C(q, q').q' + K(q)= F(q).u 
        Here, q is the minimal state vector created in the constructor. The state-space model is generated in the form r' = f(q, q', u), with r = [state[0], dstate[0], state[1], dstate[1], ..., state[n], dstate[n]], the extended state vector.

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
            * Return the variables of `sys', which has two states and two inputs and add a system to the EulerLagrange object:
            >>> from sympy.tensor.array import Array
            >>> from simupy.systems.symbolic import DynamicalSystem
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
            matrix [sympy matrix]: a matrix that needs to be checked.

        Returns:
            value [bool]: the matrix being symmetric.
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
        """Create a state space form of the system.
        """
        minimal_dstates = Matrix(self.states[1::2])
        dstates = Matrix(self.dstates[0::2])
        substitution = dict(zip(dstates, minimal_dstates))
        print(substitution)
        M_inv = self.inertia_matrix.inv()
        states_dotdot = M_inv * self.force_vector - M_inv * self.damping_matrix * minimal_dstates - M_inv * self.elastic_matrix
        states_dot = []
        for i in range(len(self.states)):
            if i%2 == 0:
                states_dot.append(self.states[i+1])
            else:
                states_dot.append( \
                    msubs(states_dotdot[i//2].copy(), substitution))
        states_dot = Array(states_dot)
        return states_dot