from nlcontrol.systems import SystemBase 

from sympy.matrices import Matrix
from sympy.tensor.array import Array
from sympy import diff, Symbol, integrate
from sympy.physics.mechanics import find_dynamicsymbols

from simupy.systems.symbolic import DynamicalSystem

import numpy as np
import itertools

class ControllerBase(SystemBase):
    def __init__(self, *args, **kwargs):
        if 'inputs' in kwargs.keys():
            inputs = kwargs['inputs']
        else:
            error_text = "[nlcontrol.systems.ControllerBase] An 'inputs=' keyword is necessary."
            raise AssertionError(error_text)
        if 'states' in kwargs.keys():
            states = kwargs['states']
        else:
            states = None
        super().__init__(states, inputs)
        self.dinputs, self.iinputs = self.__create_inputs__()

    
    def __create_inputs__(self):
        """
        Create lists of differentiated and integrated symbols of the input vector.

        Returns:
        --------
            variables : tuple 
                inputs_diff : MDimArray
                    a list of differentiated input symbols.
                inputs_int : MDimArray
                    a list of integrated input symbols.
        """
        inputs_diff = [diff(input_el, Symbol('t')) for input_el in self.inputs]
        inputs_int = [integrate(input_el, Symbol('t')) for input_el in self.inputs]
        return inputs_diff, inputs_int



class DynamicController(ControllerBase):
    def __init__(self, *args, **kwargs):
        if 'inputs' not in kwargs.keys():
            error_text = "[nlcontrol.systems.DynamicController] An 'inputs=' keyword is necessary."
            raise AssertionError(error_text)
        if 'states' not in kwargs.keys():
            error_text = "[nlcontrol.systems.DynamicController] A 'states=' keyword is necessary."
            raise AssertionError(error_text)
        super().__init__(*args, **kwargs)

        self._A = None
        self._B = None
        self._C = None
        self._f = None
        self.eta = None
        self.phi = None

        if len(args) not in (0, 6):
            error_text = '[nlcontrol.systems.DynamicController] the argument list should contain a A, B, C, f, eta, and phi matrix. If defined outside the init function, no arguments should be given.'
            raise ValueError(error_text)

        if len(args) == 6:
            self.define_controller(*args)

    def __str__(self):
        return """
        DynamicController object:
        =========================
        dz = Az - Bf(C'z) + eta(w, dw)
        u = phi(z, dz)\n
        \twith:
        \t\tA: {}
        \t\tB: {}
        \t\tC: {}
        \t\tf: {}
        \t\teta: {}
        \t\tphi: {}\n
        \tstate eq: {}
        \toutput: {}
        """.format(self.A, self.B, self.C, self.f, self.eta, self.phi, self.system.state_equation, self.system.output_equation)

    
    def define_controller(self, A, B, C, f, eta, phi):
        dim_states = self.states.shape[0]

        # Allow scalar inputs
        if np.isscalar(A):
            A = [[A]]
        if np.isscalar(B):
            B = [[B]]
        if np.isscalar(C):
            C = [[C]]
        if type(eta) is not list:
            eta = [[eta]]
        if type(phi) is not list:          
            phi = [[phi]]

        if Matrix(A).shape[1] == dim_states:
            if self.hurwitz(A):
                self.A = Matrix(A)
            else:
                error_text = '[nlcontrol.systems.DynamicController] The A matrix should be Hurwitz.'
                raise AssertionError(error_text)
        else:
            error_text = '[nlcontrol.systems.DynamicController] The number of columns of A should be equal to the number of states.'
            raise AssertionError(error_text)

        if Matrix(B).shape[0] == dim_states:
            if self.controllability(A, B):
                self.B = Matrix(B)
            else:
                error_text = '[nlcontrol.systems.DynamicController] The system is not controllable.'
                raise AssertionError(error_text)
        else:
            error_text = '[nlcontrol.systems.DynamicController] The number of rows of B should be equal to the number of states.'
            raise AssertionError(error_text)

        if Matrix(C).shape[0] == dim_states:
            if self.observability(A, C):
                self.C = Matrix(C)
            else:
                error_text = '[nlcontrol.systems.DynamicController] The system is not observable.'
                raise AssertionError(error_text)
        else:
            error_text = '[nlcontrol.systems.DynamicController] The number of rows of C should be equal to the number of states.'
            raise AssertionError(error_text)

        if callable(f):
            argument = self.C.T * Matrix(self.states)
            self.f = f(argument[0, 0])
        elif f == 0:
            self.f = 0
        else:
            error_text = '[nlcontrol.systems.DynamicController] Argument f should be a callable function or identical 0.'
            raise AssertionError(error_text)
        
        def return_dynamic_symbols(expr):
            try:
                return find_dynamicsymbols(expr)
            except:
                return set()


        if Matrix(eta).shape[0] == dim_states and Matrix(eta).shape[1] == 1:
            # Check whether the expressions only contain inputs and derivatives of the input
            dynamic_symbols_eta = [return_dynamic_symbols(eta_el) for eta_el in list(itertools.chain(*eta))]
            dynamic_symbols_eta = list(set.union(*dynamic_symbols_eta))

            inputs_list = list(self.inputs.copy())
            inputs_list.extend(self.dinputs)

            if False not in set(el in inputs_list for el in dynamic_symbols_eta):
                self.eta = Matrix(eta)
            else:
                error_text = '[nlcontrol.systems.DynamicController] Vector eta cannot contain other dynamic symbols than the inputs and its derivatives.'
                raise AssertionError(error_text) 
        else:
            error_text = '[nlcontrol.systems.DynamicController] Vector eta has an equal amount of columns as there are states. Eta has only one row.'
            raise AssertionError(error_text)

        # Check whether the expressions only contain inputs and derivatives of the input
        dynamic_symbols_phi = [return_dynamic_symbols(phi_el) for phi_el in list(itertools.chain(*phi))]
        dynamic_symbols_phi = list(set.union(*dynamic_symbols_phi))

        states_list = list(self.states.copy())
        states_list.extend(self.dstates)

        if False not in set(el in states_list for el in dynamic_symbols_phi):
            self.phi = Matrix(phi)
        else:
            error_text = '[nlcontrol.systems.DynamicController] Vector phi cannot contain other dynamic symbols than the states and its derivatives.'
            raise AssertionError(error_text)

        state_equation = Array(self.A * Matrix(self.states) + self.B * self.f + self.eta)
        output_equation = Array(self.phi)
        self.system = DynamicalSystem(state_equation=state_equation, output_equation=output_equation, state=self.states, input_=inputs_list)


    def controllability(self, A, B):
        A = np.array(A)
        B = np.array(B)
        p = np.linalg.matrix_rank(A)
        zeta = None
        for i in range(p):
            A_to_i_times_B = np.linalg.matrix_power(A, i).dot(B)
            if zeta is None:
                zeta = A_to_i_times_B
            else:
                zeta = np.append(zeta, A_to_i_times_B, axis=1)
        if np.linalg.matrix_rank(zeta) == p:
            return True
        else:
            return False
        

    def observability(self, A, C):
        A = np.array(A)
        C = np.array(C).T
        p = np.linalg.matrix_rank(A)
        Q = None
        for i in range(p):
            C_times_A_to_i = C.dot(np.linalg.matrix_power(A, i))
            if Q is None:
                Q = C_times_A_to_i
            else:
                Q = np.append(Q, C_times_A_to_i, axis=0)
        if np.linalg.matrix_rank(Q) == p:
            return True
        else:
            return False


    def hurwitz(self, matrix):
        matrix = np.array(matrix)
        eig,_ = np.linalg.eig(matrix)
        check_eig = [True if eig < 0  else False for eig in np.real(eig)]
        if False in check_eig:
            return False
        else: 
            return True
        