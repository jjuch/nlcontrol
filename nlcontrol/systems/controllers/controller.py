from nlcontrol.systems import SystemBase 

from sympy.matrices import Matrix
from sympy.tensor.array import Array
from sympy import diff, Symbol, integrate
from sympy.physics.mechanics import find_dynamicsymbols, msubs
from sympy.core.function import Derivative

from simupy.systems.symbolic import DynamicalSystem

import numpy as np
import itertools

class ControllerBase(SystemBase):
    """
    ControllerBase(states, inputs, sys=None)

    Returns a base structure for a controller with outputs, optional inputs, and optional states. The controller is an instance of a SystemBase, which is defined by it state equations (optional):
        diff(x(t), t) = h(x(t), u(t), t)
    with x(t) the state vector, u(t) the input vector and t the time in seconds. Next, the output is given by the output equation:
        y(t) = g(x(t), u(t), t)

    Parameters:
    -----------
        states : string or array-like
            if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols.
        inputs : string or array-like
            if `inputs` is a string, it is a comma-separated listing of the input names. If `inputs` is array-like it contains the inputs as sympy's dynamic symbols.
        system : simupy's DynamicalSystem object (simupy.systems.symbolic), optional
            the object containing output and state equations, default: None.

    Examples:
    ---------
        * Statefull controller with one state, one input, and one output:
            >>> from simupy.systems.symbolic import MemorylessSystem, DynamicalSystem
            >>> from sympy.tensor.array import Array
            >>> st = 'z'
            >>> inp = 'w'
            >>> contr = ControllerBase(states=st, inputs=inp)
            >>> z, zdot, w = contr.create_variables()
            >>> contr.system = DynamicalSystem(state_equation=Array([-z + w]), state=z, output_equation=z, input_=w)

        * Statefull controller with two states, one input, and two outputs:
            >>> st = 'z1, z2'
            >>> inp = 'w'
            >>> contr = ControllerBase(states=st, inputs=inp)
            >>> z1, z2, z1dot, z2dot, w = contr.create_variables()
            >>> contr.system = DynamicalSystem(state_equation=Array([-z1 + z2**2 + w, -z2 + 0.5 * z1]), state=Array([z1, z2]), output_equation=Array([z1 * z2, z2]), input_=w)

        * Stateless controller with one input:
            >>> st = None
            >>> inp = 'w'
            >>> contr = ControllerBase(states=st, inputs=inp)
            >>> w = contr.create_variables()
            >>> contr.system = MemorylessSystem(input_=Array([w]), output_equation= Array([5 * w]))

        * Create a copy a ControllerBase object `contr' and linearize around the working point of state [0, 0] and working point of input 0 and simulate:
            >>> new_contr = ControllerBase(states=contr.states, inputs=contr.inputs, sys=contr.system)
            >>> new_contr_lin = new_contr.linearize([0, 0], 0)
            >>> new_contr_lin.simulation(10)
    """
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
        if 'sys' in kwargs.keys():
            sys = kwargs['sys']
        else:
            sys = None
        super().__init__(states, inputs, sys=sys)
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

    
    def series(self, contr_append):
        """
            A controller is generated which is the result of a serial connection of two controllers. The outputs of this object are connected to the inputs of the appended system and a new controller is achieved which has the inputs of the current system and the outputs of the appended system. Notice that the dimensions of the output of the current system should be equal to the dimension of the input of the appended system.

            Parameters:
            -----------
                contr_append : ControllerBase object
                    the controller that is placed in a serial configuration. `contr_append' follows the current system.

            Returns:
            --------
                A ControllerBase object with the serial system's equations.

            Examples:
            ---------
                * Place `contr1' behind `contr2' in a serial configuration and show the inputs, states, state equations and output equations:
                >>> series_sys = contr1.series(contr2)
                >>> print('inputs: ', series_sys.system.input_)
                >>> print('States: ', series_sys.system.state)
                >>> print('State eq's: ', series_sys.system.state_equation)
                >>> print('Output eq's: ', series_sys.system.output_equation)
        """
        series_system = super().series(contr_append)
        return ControllerBase(inputs=series_system.inputs, states=series_system.states, sys=series_system.system)

    
    def parallel(self, contr_append):
        """
            A controller is generated which is the result of a parallel connection of two controllers. The inputs of this object are connected to the system that is placed in parallel and a new system is achieved with the output the sum of the outputs of both systems in parallel. Notice that the dimensions of the inputs and the outputs of both systems should be equal.

            Parameters:
            -----------
                contr_append : ControllerBase object
                    the controller that is added in parallel.

            Returns:
            --------
                A ControllerBase object with the parallel system's equations.

            Examples:
            ---------
                * Place `contr2' in parallel with `contr1' and show the inputs, states, state equations and output equations:
                >>> parallel_sys = contr1.parallel(contr2)
                >>> print('inputs: ', parallel_sys.system.input_)
                >>> print('States: ', parallel_sys.system.state)
                >>> print('State eq's: ', parallel_sys.system.state_equation)
                >>> print('Output eq's: ', parallel_sys.system.output_equation)
        """
        parallel_system = super().parallel(contr_append)
        return ControllerBase(inputs=parallel_system.inputs, states=parallel_system.states, sys=parallel_system.system)



class DynamicController(ControllerBase):
    """
    DynamicController(states=None, inputs=None, sys=None)

    The DynamicController object is based on the ControllerBase class. A dynamic controller is defined by the following differential equations:
        diff(z(t), t) = A.z(t) - B.f(sigma(t)) + eta(w(t), diff(w(t),t))
        sigma(t) = C'.z
        u0 = phi(z(t), diff(z(t), t))
    
    with z(t) the state vector, w(t) the input vector and t the time in seconds. the symbol ' refers to the transpose. 
    Conditions:
        * A is Hurwitz,
        * (A, B) should be controllable, 
        * (A, C) is observable,
        * rank(B) = rang (C) = s <= n, with s the dimension of sigma, and n the number of states. 

    Parameters:
    -----------
        states : string or array-like
            if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols.
        inputs : string or array-like
            if `inputs` is a string, it is a comma-separated listing of the input names. If `inputs` is array-like it contains the inputs as sympy's dynamic symbols. Do not provide the derivatives as these will be added automatically.
        system : simupy's DynamicalSystem object (simupy.systems.symbolic), optional
            the object containing output and state equations, default: None.

    Examples:
    ---------
        * Statefull controller with two states, one input, and two outputs:
            >>> inp = 'w'
            >>> st = 'z1, z2'
            >>> contr = DynamicController(states=st, inputs=inp)
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
            >>> print(contr)

    """
    def __init__(self, *args, **kwargs):
        if 'inputs' not in kwargs.keys():
            error_text = "[nlcontrol.systems.DynamicController] An 'inputs=' keyword is necessary."
            raise AssertionError(error_text)
        if 'states' not in kwargs.keys():
            error_text = "[nlcontrol.systems.DynamicController] A 'states=' keyword is necessary."
            raise AssertionError(error_text)
        super().__init__(*args, **kwargs)
        
        self.minimal_inputs = self.inputs
        self.inputs = Array([val for pair in zip(self.inputs, self.dinputs) for val in pair])
        
        self.A = None
        self.B = None
        self.C = None
        self.f = None
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
        """
        Define the Dynamic controller given by the differential equations:
            diff(z(t), t) = A.z(t) - B.f(sigma(t)) + eta(w(t), diff(w(t),t))
            sigma(t) = C'.z
            u0 = phi(z(t), diff(z(t), t))
    
        with z(t) the state vector, w(t) the input vector and t the time in seconds. the symbol ' refers to the transpose. 
        Conditions:
            * A is Hurwitz,
            * (A, B) should be controllable, 
            * (A, C) is observable,
            * rank(B) = rang (C) = s <= n, with s the dimension of sigma, and n the number of states.

        HINT: use create_variables() for an easy notation of the equations.

        Parameters:
        -----------
            A : array-like
                Hurwitz matrix. Size: n x n
            B : array-like
                In combination with matrix A, the controllability is checked. The linear definition can be used. Size: s x n
            C : array-like
                In combination with matrix A, the observability is checked. The linear definition can be used. Size: n x 1
            f : callable (lambda-function)
                A (non)linear lambda function with argument sigma, which equals C'.z.
            eta : array-like
                The (non)linear relation between the inputs plus its derivatives to the change in state. Size: n x 1
            phi : array-like
                The (non)linear output equation. The equations should only contain states and its derivatives. Size: n x 1

        Examples:
        ---------
            See DyncamicController object.
        """
        dim_states = self.states.shape[0]

        # Allow scalar inputs
        if np.isscalar(A):
            A = [[A]]
        if np.isscalar(B):
            B = [[B]]
        if np.isscalar(C):
            C = [[C]]
        if type(eta) not in (list, Matrix):
            eta = [[eta]]
        if type(phi) not in (list, Matrix):          
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
            if self.controllability_linear(A, B):
                self.B = Matrix(B)
            else:
                error_text = '[nlcontrol.systems.DynamicController] The system is not controllable.'
                raise AssertionError(error_text)
        else:
            error_text = '[nlcontrol.systems.DynamicController] The number of rows of B should be equal to the number of states.'
            raise AssertionError(error_text)
        
        if Matrix(C).shape[0] == dim_states:
            if self.observability_linear(A, C):
                self.C = Matrix(C)
            else:
                error_text = '[nlcontrol.systems.DynamicController] The system is not observable.'
                raise AssertionError(error_text)
        else:
            error_text = '[nlcontrol.systems.DynamicController] The number of rows of C should be equal to the number of states.'
            raise AssertionError(error_text)

        if type(f) is not Matrix:
            if callable(f):
                argument = self.C.T * Matrix(self.states)
                #TODO: make an array of f
                self.f = f(argument[0, 0])
            elif f == 0:
                self.f = 0
            else:
                error_text = '[nlcontrol.systems.DynamicController] Argument f should be a callable function or identical 0.'
                raise AssertionError(error_text)
        else:
            self.f = f
        

        def return_dynamic_symbols(expr):
            try:
                return find_dynamicsymbols(expr)
            except:
                return set()
        
        if Matrix(eta).shape[0] == dim_states and Matrix(eta).shape[1] == 1:
            # Check whether the expressions only contain inputs
            if type(eta) is Matrix:
                dynamic_symbols_eta = [return_dynamic_symbols(eta_el[0]) for eta_el in eta.tolist()]
            else:
                dynamic_symbols_eta = [return_dynamic_symbols(eta_el) for eta_el in list(itertools.chain(*eta))]
            dynamic_symbols_eta = set.union(*dynamic_symbols_eta)

            if dynamic_symbols_eta <= (
                set(self.inputs)
            ):
                self.eta = Matrix(eta)
            else:
                error_text = '[nlcontrol.systems.DynamicController] Vector eta cannot contain other dynamic symbols than the inputs.'
                raise AssertionError(error_text) 
        else:
            error_text = '[nlcontrol.systems.DynamicController] Vector eta has an equal amount of columns as there are states. Eta has only one row.'
            raise AssertionError(error_text)

        # Check whether the expressions only contain inputs and derivatives of the input
        if type(phi) is Matrix:
            dynamic_symbols_phi = [return_dynamic_symbols(phi_el[0]) for phi_el in phi.tolist()]
        else:
            dynamic_symbols_phi = [return_dynamic_symbols(phi_el) for phi_el in list(itertools.chain(*phi))]
        dynamic_symbols_phi = set.union(*dynamic_symbols_phi)

        if dynamic_symbols_phi <= (
            set(self.states) | set(self.dstates)
        ):
            self.phi = Matrix(phi)
        else:
            error_text = '[nlcontrol.systems.DynamicController] Vector phi cannot contain other dynamic symbols than the states and its derivatives.'
            raise AssertionError(error_text)

        state_equation = Array(self.A * Matrix(self.states) - self.B * self.f + self.eta)
        output_equation = Array(self.phi)
        diff_states = []
        for state in self.states:
            diff_states.append(Derivative(state, Symbol('t')))
        substitutions = dict(zip(diff_states, state_equation))
        # print('Subs: ', substitutions)
        output_equation = msubs(output_equation, substitutions)
        self.system = DynamicalSystem(state_equation=state_equation, output_equation=output_equation, state=self.states, input_=self.inputs)


    def controllability_linear(self, A, B):
        """
        Controllability check of two matrices using the Kalman rank condition for time-invariant linear systems [1].

        Parameters:
        -----------
            A : array-like
                Size: n x n
            B : array-like
                Size: s x n

        [1]. R.E. Kalman, "On the general theory of control systems", IFAC Proc., vol. 1(1), pp. 491-502, 1960. doi.10.1016/S1474-6670(17)70094-8.
        """
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
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
        

    def observability_linear(self, A, C):
        """
        Observability check of two matrices using the Kalman rank condition for time-invariant linear systems [1].

        Parameters:
        -----------
            A : array-like
                Size: n x n
            C : array-like
                Size: n x 1

        [1]. R.E. Kalman, "On the general theory of control systems", IFAC Proc., vol. 1(1), pp. 491-502, 1960. doi.10.1016/S1474-6670(17)70094-8.
        """
        A = np.array(A, dtype=float)
        C = np.array(C, dtype=float).T
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
        """
        Check whether a time-invariant matrix is Hurwitz. The real part of the eigenvalues should be smaller than zero.

        Parameters:
        -----------
            matrix: array-like
                A square matrix.
        """
        matrix = np.array(matrix, dtype=float)
        eig,_ = np.linalg.eig(matrix)
        check_eig = [True if eig < 0  else False for eig in np.real(eig)]
        if False in check_eig:
            return False
        else: 
            return True
        