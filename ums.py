import numpy as np
import matplotlib.pyplot as plt

from sympy.physics.mechanics import dynamicsymbols, msubs
from sympy import symbols, cos, sin
from sympy.matrices import Matrix, ones
from sympy.tensor.array import Array
from sympy import diff

from simupy.systems.symbolic import DynamicalSystem
from simupy.block_diagram import BlockDiagram

class UMS():
    """
    A class to simulate Underactuated Mechanical Systems.

    Attributes
    ----------
    M: Inertia matrix, the matrix is positive definite symmetric.
    C: Coriolis/Centrifugal matrix.
    K: Gravity term.
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
            :states [str]: Position state variables
            :inputs [str]: input variables
        """
        self.M = None
        self.C = None
        self.K = None
        self.F = None
        self.states = Matrix(dynamicsymbols(states))
        self.dstates = Matrix(dynamicsymbols(states, 1))
        if (',' in inputs):
            self.inputs = Matrix(dynamicsymbols(inputs))
        else:
            self.inputs = Matrix([dynamicsymbols(inputs)])
        self.x = None
        self.xdot = None
        self.sys = None

    def createVariables(self) -> tuple:
        '''
        Returns a tuple with all variables. First the states are given, next the derivative of the states, and finally the inputs.

        Returns:
        --------
        :Variables [tuple]: all variables as described above
        '''
        var_list_states = self.dstates.row_insert(0, self.states)
        var_list = self.inputs.row_insert(0, var_list_states) 
        return tuple(var_list)    

    def define_system(self, M, C, K, F) -> bool:
        """Define the UMS system using the differential equation representation:
            M(q).q'' + C(q, q').q' + K(q)= F(q).u 
        Here, q is the state vector created in the constructor. A state-space model is generated in the form r' = f(q, q', u), with r = [state[0], dstate[0], state[1], dstate[1], ..., state[n], dstate[n]].

        Parameters:
        -----------
        :M [list]: Inertia matrix, the matrix is positive definite symmetric. Size: n x n
        :C [list]: Coriolis/Centrifugal matrix. Size: m x n
        :K [list]: Gravity term. Size: n x 1
        :F [list]: External forces, non-square matrix. Size: n x 1

        Output:
        -------
        :bool: success status
        """
        # Transform to sympy matrices
        M_mat = Matrix(M)
        C_mat = Matrix(C)
        K_mat = Matrix(K)
        F_mat = Matrix(F)

        length_states = len(self.states)
        # M should be symmetric
        if self.check_symmetry(M_mat):
            # M should have dimensionality n
            if M_mat.shape[0] == length_states:
                self.M = M_mat
            else:
                print('Error: Matrix M should be squared.')
                return False
        else:
            print('Error: Matrix M should be symmetric')
            return False

        # Matrix C should have the dimension m x n
        if C_mat.shape[1] == length_states:
            self.C = C_mat
        else:
            print('Error: Matrix C should have a row length equal to the number of states.')
            return False

        # Matrix K should have the dimension 1 x n
        if K_mat.shape[0] == length_states:
            self.K = K_mat
        else:
            print('Error: Matrix K should have the same length as the length of the state vector.')
            return False

        # Matrix F should have the dimension 1 x n
        if F_mat.shape[0] == length_states:
            self.F = F_mat
        else:
            print('Error: Matrix F should have the same length as the length of the state vector.')
            return False

        self.x, self.xdot = self.create_statespace()

        self.sys = DynamicalSystem(state_equation=self.xdot, state=self.x, input_=self.inputs)
        return True
        

    
    def check_symmetry(self, matrix) -> bool:
        """Check if matrix (sympy) is symmetric. Returns a bool.
        Parameter:
        ----------
        :matrix: a matrix that needs to be checked
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
        """Create a state space form of the system. Returns a tuple.
        Outputs:
        --------
        :r: array of states.
        :r_dot: array of solutions for the derivative of states.
        """
        M_inv = self.M.inv()
        x_dotdot = M_inv * self.F - M_inv * self.C * self.dstates - M_inv * self.K
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

    def simulate_system(self, initial_conditions, tspan, system=None, show: bool=False) -> object:
        """Simulate the system for an initial condition for a predefined length. There is also an option to show figures.
        Parameters:
        -----------
        :initial_conditions: array of initial conditions for each state.
        :tspan: the length of the simulation in seconds.
        :system: (optional, default self.sys) Simupy DynamicalSystem object.
        :show: (optional, default False) boolean to indicate if plots need to be shown.
        """
        if system is None:
            system = self.sys
        
        BD = BlockDiagram(system)
        system.initial_condition = initial_conditions
        res = BD.simulate(tspan)
        
        x = res.x[:, 0]
        theta = res.x[:, 2]

        if show:
            plt.figure()
            ObjectLines = plt.plot(res.t, x, res.t, theta)
            plt.legend(iter(ObjectLines), ['x', 'theta'])
            plt.show()

            plt.figure()
            ObjectLines = plt.plot(res.x[:, 0], res.x[:, 1])
            plt.show()

            plt.figure()
            ObjectLines = plt.plot(res.x[:, 2], res.x[:, 3])
            plt.show()

        return res


if __name__ == '__main__':
    states = 'x, theta'
    inputs = 'L'

    ums_noG = UMS(states, inputs)
    ums_G = UMS(states, inputs)
    
    # No gravity
    e = 0.5
    M = Matrix([[1, e*cos(ums_noG.states[1])], [e*cos(ums_noG.states[1]), 1]])
    C = Matrix([[0, (-1)*e*ums_noG.dstates[1]*sin(ums_noG.states[1])], [0, 0]])
    K = Matrix([[ums_noG.states[0]],[0]])
    F = Matrix([[0], [ums_noG.inputs]])

    ums_noG.define_system(M, C, K, F)
    ums_noG.simulate_system([1, 0, np.pi/4, 0], 40, show=True)
    # ums_noG.simulate_system([0.1, 0, np.pi, 0], 40)

    e = 0.5
    G = 1
    M = Matrix([[1, e*cos(ums_G.states[1])], [e*cos(ums_G.states[1]), 1]])
    C = Matrix([[0, (-1)*e*ums_G.dstates[1]*sin(ums_G.states[1])], [0, 0]])
    K = Matrix([[ums_G.states[0]],[G*sin(ums_G.states[1])]])
    F = Matrix([[0], [ums_G.inputs]])

    ums_G.define_system(M, C, K, F)
    ums_G.simulate_system([1, 0, np.pi/4, 0], 40, show=True)

    print(ums_G.xdot)