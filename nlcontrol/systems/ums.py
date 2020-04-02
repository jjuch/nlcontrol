import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

from sympy.matrices import Matrix
from sympy.physics.mechanics import dynamicsymbols

from simupy.block_diagram import BlockDiagram
from simupy.systems.symbolic import DynamicalSystem

from nlcontrol.systems import EulerLagrange

class UMS(EulerLagrange):
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
        The underactuated mechanical systems class uses the Euler-Lagrange notation for its system definition. An underactuated system is described with the following differential equation: 
            M(q).q'' + C(q, q').q' + K(q)= F(q).u

        Parameters:
            :states [str]: Position state variables
            :inputs [str]: input variables
        """
        super().__init__(states, inputs)


    def define_system(self, M, C, K, F) -> bool:
        """Define the UMS system using the differential equation representation:
            M(q).q'' + C(q, q').q' + K(q)= F(q).u 
        Here, q is the state vector created in the constructor. A state-space model is generated in the form r' = f(q, q', u), with r = [state[0], dstate[0], state[1], dstate[1], ..., state[n], dstate[n]].

        HINT: use createVariables() for an easy notation of state[i] and dstate[i].

        Parameters:
            M [list]: Inertia matrix, the matrix is positive definite symmetric. Size: n x n
            C [list]: Damping matrix. Size: m x n
            K [list]: Elastic matrix. Size: n x 1
            F [list]: External forces, non-square matrix. Size: n x 1

        Returns:
            value [bool]: success status
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
                # self.M = M_mat
                self.inertia_matrix = M_mat
            else:
                print('Error: Matrix M should be squared.')
                return False
        else:
            print('Error: Matrix M should be symmetric')
            return False

        # Matrix C should have the dimension m x n
        if C_mat.shape[1] == length_states:
            # self.C = C_mat
            self.damping_matrix = C_mat
        else:
            print('Error: Matrix C should have a row length equal to the number of states.')
            return False

        # Matrix K should have the dimension 1 x n
        if K_mat.shape[0] == length_states:
            # self.K = K_mat
            self.elastic_matrix = K_mat
        else:
            print('Error: Matrix K should have the same length as the length of the state vector.')
            return False

        # Matrix F should have the dimension 1 x n
        if F_mat.shape[0] == length_states:
            self._F = F_mat
        else:
            print('Error: Matrix F should have the same length as the length of the state vector.')
            return False
        self.x, self.xdot = self.create_statespace()

        self.sys = DynamicalSystem(state_equation=self.xdot, state=self.x, input_=self.inputs)
        return True
    

    def simulate_system(self, initial_conditions, tspan, system=None, show: bool=False, nsteps:int=None) -> object:
        """Simulate the system for an initial condition for a predefined length. There is also an option to show figures.
        Parameters:
        -----------
        :initial_conditions: array of initial conditions for each state.
        :tspan: the length of the simulation in seconds.
        :system: (optional, default self.sys) Simupy DynamicalSystem object.
        :show: (optional, default False) boolean to indicate if plots need to be shown.
        :nsteps: (optional, default None) integer, steps per integrator step, scipy default is 500.
        """
        if system is None:
            system = self.sys
        
        BD = BlockDiagram(system)
        system.initial_condition = initial_conditions
        if (nsteps != None):
            INTEGRATOR_OPTIONS = {
                'name': 'dopri5',
                'rtol': 1e-6,
                'atol': 1e-12,
                'nsteps': nsteps,
                'max_step': 0.0
            }
            res = BD.simulate(tspan, integrator_options=INTEGRATOR_OPTIONS)
        else:
            res = BD.simulate(tspan)
        
        x = res.x[:, 0]
        theta = res.x[:, 2]

        plt.figure()
        ObjectLines = plt.plot(res.t, x, res.t, theta)
        plt.legend(iter(ObjectLines), [el for el in tuple(self.states)])
        plt.title('states versus time')
        plt.xlabel('time (s)')
        plt.show()

        if show:
            plt.figure()
            ObjectLines = plt.plot(res.x[:, 0], res.x[:, 1])
            plt.title(r'$q_1$ vs $\displaystyle\frac{dq_1}{dt}$')
            plt.xlabel(r'$q_1$')
            plt.ylabel(r'$\dot{q_1}$')
            plt.show()

            plt.figure()
            ObjectLines = plt.plot(res.x[:, 2], res.x[:, 3])
            plt.title(r'$q_2$ vs $\displaystyle\frac{dq_2}{dt}$')
            plt.xlabel(r'$q_2$')
            plt.ylabel(r'$\dot{q_2}$')
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