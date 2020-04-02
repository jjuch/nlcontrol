from sympy.physics.mechanics import dynamicsymbols, msubs
from sympy import symbols, cos, sin
from sympy.matrices import Matrix, ones
from sympy.tensor.array import Array
from sympy import diff, Symbol

from simupy.systems.symbolic import DynamicalSystem

from nlcontrol.systems import SystemBase

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


    def __init__(self, states:str, inputs: str):
        """
        The underactuated mechanical systems class uses four matrices to describe the equations of motion. An underactuated system is described with the following differential equation: 
            M(q).q'' + C(q, q').q' + K(q)= F(q).u

        Parameters:
            states [str]: Position state variables. The variables are separated by ','.
            inputs [str]: input variables. The variables are separated by ','.
        """
        super().__init__(states, inputs)
        self._M = None
        self._C = None
        self._K = None
        self._F = None
        self.x = None
        self.xdot = None

    @property
    def inertia_matrix(self) -> Matrix:
        return self._M

    @inertia_matrix.setter
    def inertia_matrix(self, matrix:Matrix) -> bool:
        self._M = matrix
        return True

    @property
    def damping_matrix(self) -> Matrix:
        return self._C

    @damping_matrix.setter
    def damping_matrix(self, matrix:Matrix) -> bool:
        self._C = matrix
        return True

    @property
    def elastic_matrix(self) -> Matrix:
        return self._K

    @elastic_matrix.setter
    def elastic_matrix(self, matrix:Matrix) -> bool:
        self._K = matrix
        return True

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

    def create_statespace(self) -> tuple:
        """Create a state space form of the system.

        Returns:
            r: array of states.
            r_dot: array of solutions for the derivative of states.
        """
        M_inv = self.inertia_matrix.inv()
        x_dotdot = M_inv * self._F - M_inv * self.damping_matrix * self.dstates - M_inv * self.elastic_matrix
        r_dot = []
        n = len(self.states)
        r = Array(dynamicsymbols('r1:'+str(n*2+1)))
        for i in range(2*n):
            if i%2 == 0:
                r_dot.append(r[i+1])
            else:
                x_dotdot_temp = x_dotdot[i//2].copy()
                for k in range(n):
                    # x_dotdot_temp.subs(self.states[k], r[2*k])
                    x_dotdot_temp = msubs(x_dotdot_temp, {self.states[k]: r[2*k]})
                    x_dotdot_temp = msubs(x_dotdot_temp, {self.dstates[k]: r[2*k+1]})            
                r_dot.append(x_dotdot_temp)
        r = Array(r)
        r_dot = Array(r_dot)
        # r = np.c_[r]
        # r_dot = np.c_[r_dot]
        return r, r_dot


    def linearize(self, working_point) -> tuple:
        '''linearize the Euler-Lagrange formulation around a working point.

        Parameters:
            working_point [list]: a list with working point values for speeds and positions.

        Returns:
            r_dot_lin [Array]: linearized system.
            sys_lin [DynamicalSystem]: linearized system.        
        '''
        if (len(working_point) != len(self.x)):
            print('Error: Matrix K should have the same length as the length of the state vector.')
        else:
            subs = {r: working_point[index] for (index, r) in enumerate(self.x)}
            r_dot_lin = []
            for k in range(len(self.xdot)):
                linearized_term = 0
                for j in range(len(self.x)):
                    linearized_term += msubs(diff(self.xdot[k], self.x[j]), subs) * (self.x[j] - subs[self.x[j]])
                linearized_term += msubs(self.xdot[k], subs)
                r_dot_lin.append(linearized_term)
            r_dot_lin = Array(r_dot_lin)
            sys_lin = DynamicalSystem(state_equation=r_dot_lin, state=self.x, input_=self.inputs)
            return r_dot_lin, sys_lin