from sympy import Symbol
from sympy.matrices import Matrix, eye, BlockMatrix, zeros
from sympy.tensor.array import Array
from sympy.physics.mechanics import dynamicsymbols, msubs
from sympy.core.function import Derivative

from nlcontrol.systems.controllers import ControllerBase

from simupy.systems.symbolic import DynamicalSystem

import numpy as np
import itertools

__all__ = ["DynamicController", "EL_circ"]

class DynamicController(ControllerBase):
    """
    DynamicController(states=None, inputs=None, sys=None, name="EL controller")

    The DynamicController object is based on the ControllerBase class. A dynamic controller is defined by the following differential equations:

    .. math::
        \\frac{dz(t)}{dt} = A.z(t) - B.f(\\sigma(t)) + \\eta\\left(w(t), \\frac{dw(t)}{dt}\\right)
    
    .. math::
        \\sigma(t) = C'.z
    
    .. math::
        u_0 = \\phi\\left(z(t), \\frac{dz(t)}{dt}\\right)
    
    with z(t) the state vector, w(t) the input vector and t the time in seconds. the symbol ' refers to the transpose. 
    
    **Conditions:**

        * A is Hurwitz,
        * (A, B) should be controllable, 
        * (A, C) is observable,
        * rank(B) = rang (C) = s <= n, with s the dimension of sigma, and n the number of states.

    More info on the controller can be found in [1, 2].

    Parameters
    -----------
    states : string or array-like
        if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols.
    inputs : string or array-like
        if `inputs` is a string, it is a comma-separated listing of the input names. If `inputs` is array-like it contains the inputs as sympy's dynamic symbols. Do not provide the derivatives as these will be added automatically.
    system : simupy's DynamicalSystem object (simupy.systems.symbolic), optional
        the object containing output and state equations, default: None.
    name : string
        give the system a custom name which will be shown in the block scheme, default: 'EL controller'.

    Examples
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

    References
    -----------
    [1] L. Luyckx, The nonlinear control of underactuated mechanical systems. PhD thesis, UGent, Ghent, Belgium, 5 2006.

    [2] M. Loccufier, "Stabilization and set-point regulation of underactuated mechanical systems", Journal of Physics: Conference Series, 2016, vol. 744, no. 1, p.012065.

    """
    def __init__(self, *args, **kwargs):
        if 'inputs' not in kwargs.keys():
            error_text = "[nlcontrol.systems.DynamicController] An 'inputs=' keyword is necessary."
            raise AssertionError(error_text)
        else:
            inputs = kwargs['inputs']
        if 'states' not in kwargs.keys():
            error_text = "[nlcontrol.systems.DynamicController] A 'states=' keyword is necessary."
            raise AssertionError(error_text)
        else:
            states = kwargs['states']
        if 'name' not in kwargs.keys():
            kwargs['name'] = "EL controller"
        super().__init__(states, inputs, *args, **kwargs)
        
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

        .. math::
            \\frac{dz(t)}{dt} = A.z(t) - B.f(\\sigma(t)) + \\eta\\left(w(t), \\frac{dw(t)}{dt}\\right)
        
        .. math::
            \\sigma(t) = C'.z
        
        .. math::
            u_0 = \\phi\\left(z(t), \\frac{dz(t)}{dt}\\right)
    
        with z(t) the state vector, w(t) the input vector and t the time in seconds. the symbol ' refers to the transpose. 
        Conditions:
            * A is Hurwitz,
            * (A, B) should be controllable, 
            * (A, C) is observable,
            * rank(B) = rang (C) = s <= n, with s the dimension of sigma, and n the number of states.

        **HINT:** use create_variables() for an easy notation of the equations.

        Parameters
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

        **Reference:**

        [1]. R.E. Kalman, "On the general theory of control systems", IFAC Proc., vol. 1(1), pp. 491-502, 1960. doi.10.1016/S1474-6670(17)70094-8.

        Parameters
        -----------
        A : array-like
            Size: n x n
        B : array-like
            Size: s x n
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

        **Reference:**
        
        [1] R.E. Kalman, "On the general theory of control systems", IFAC Proc., vol. 1(1), pp. 491-502, 1960. doi.10.1016/S1474-6670(17)70094-8.

        Parameters
        -----------
        A : array-like
            Size: n x n
        C : array-like
            Size: n x 1
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

        Parameters
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
        

class EL_circ(DynamicController):
    """
    The EL_circ creates an Euler-Lagrange controller that is based on on the DynamicController class. It's stability and structure follows from the proof in [1] and is based on the circle criterium. This does not return a general Euler-Lagrange controller, but rather a special case. The control equation is:

    .. math::
        D0.p'' + C0.p' + K0.p + C1.f(C1^T.p) + N0.w' = 0

    The apostrophe represents a time derivative, :math:`.^T` is the transpose of the matrix. 

    The output equation is:

    .. math::
        {NA}^T.D0^{-1}.K0^{-1}.D0.K0.p - {NB}^T.D0^{-1}.K0^{-1}.D0.K0.p'
    
    More info on the controller can be found in [1, 2]. Here, the parameters are chosen to be

        * :math:`\\bar{\\gamma} = 0`
        * :math:`\\bar{\\alpha} = I`
    
    with I the identity matrix.

    Parameters
    -----------
    D0 : matrix-like
        inertia matrix with numerical values. The matrix should be positive definite and symmetric.
    C0 : matrix-like
        linear damping matrix with numerical values. The matrix should be positive definite and symmetric.
    K0 : matrix-like
        linear stiffness matrix with numerical values. The matrix should be positive definite and symmetric.
    C1 : matrix-like
        nonlinear function's constant matrix with numerical values.
    f : matrix-like
        nonlinear functions containing lambda functions.
    NA : matrix-like
        the numerical multipliers of the position feedback variables.
    NB : matrix-like
        the numerical multipliers of the velocity feedback variables.
    nonlinearity_type : string
        the nonlinear part acts on the state or the derivative of the state of the dynamic controller. The only options are `stiffness' and `damping'.

    Attributes
    -----------
    D0 : inertia_matrix
        Inertia forces.
    C0 : damping_matrix
        Damping and Coriolis forces.
    K0 : stiffness_matrix
        Elastic en centrifugal forces.
    C1 : nonlinear_coefficient_matrix
        Coëfficient of the nonlinear functions.
    type : nl_stiffness
        A boolean indicating whether a nonlinear stiffness (True) or damping (False) is present.
    nl : nonlinear_fcts
        Nonlinear functions of the controller.
    nl_call : nonlinear_fcts_callable
        Callable lambda functions of the nonlinear functions.
    NA : gain_inputs
        Coëfficients of the position inputs.
    NB : gain_dinputs
        Coëfficients of the velocity inputs.
    inputs : sympy array of dynamicsymbols
        input variables.
    dinputs : sympy array of dynamicsymbols
        derivative of the input array
    states : sympy array of dynamicsymbols
        state variables.
        
        
    Examples
    ---------
    * An Euler-Lagrange controller with two states, the input is the minimal state of a BasicSystem `sys' and the nonlinearity is acting on the position variable of the Euler-Lagrange controller's state:
        >>> from sympy import atan
        >>> D0 = [[1, 0], [0, 1.5]]
        >>> C0 = [[25, 0], [0, 35]]
        >>> K0 = [[1, 0], [0, 1]]
        >>> C1 = [[0.5, 0], [0, 0.5]]
        >>> s_star = 0.01
        >>> f0 = 10
        >>> f1 = 1
        >>> f2 = (f0 - f1)*s_star
        >>> func = lambda x: f1 * x + f2 * atan((f0 - f1)/f2 * x)
        >>> f = [[func], [func]]
        >>> NA = [[0, 0], [0, 0]]
        >>> NB = [[3, 0], [2.5, 0]]
        >>> contr = EulerLagrangeController(D0, C0, K0, C1, f, NA, NB, sys.minimal_states, nonlinearity_type='stiffness')
    
    References
    -----------
    [1] L. Luyckx, The nonlinear control of underactuated mechanical systems. PhD thesis, UGent, Ghent, Belgium, 5 2006.

    [2] M. Loccufier, "Stabilization and set-point regulation of underactuated mechanical systems", Journal of Physics: Conference Series, 2016, vol. 744, no. 1, p.012065.

    """
    def __init__(self, D0, C0, K0, C1, f, NA, NB, inputs, nonlinearity_type='stiffness'):
        self._D0 = None
        self._C0 = None
        self._K0 = None
        self._C1 = None
        if nonlinearity_type == 'stiffness':
            self.nl_stiffness = True
        elif nonlinearity_type == 'damping':
            self.nl_stiffness = False
        else:
            error_text = "[EulerLagrangeController.init] The keyword 'nonlinearity_type' should be a string which is equal to 'stiffness' or 'damping'."
            raise ValueError(error_text)

        self._nl = None
        self._nl_call = None
        self._NA = None
        self._NB = None
        self.inputs = inputs
        self.dinputs, _ = self.__create_inputs__()

        if type(D0) in (float, int):
            D0 = [[D0]]
            C0 = [[C0]]
            K0 = [[K0]]
            C1 = [[C1]]
            f = [[f]]
        if len(self.inputs) == 1:
            NA = [[NA]]
            NB = [[NB]]
        self.inertia_matrix = Matrix(D0)
        self.minimal_states = self.create_states(len(D0))
        self.minimal_dstates = self.create_states(len(D0), level=1)
        self.states = self.create_states(len(D0) * 2)
        self.damping_matrix = Matrix(C0)
        self.stiffness_matrix = Matrix(K0)
        self.nonlinear_coefficient_matrix = Matrix(C1)
        self.nonlinear_fcts = f
        self.gain_inputs = Matrix(NA)
        self.gain_dinputs = Matrix(NB)

        # Create system
        super().__init__(states = self.states, inputs = self.inputs)
        A, B, C, f, eta, phi  = self.convert_to_dynamic_controller()
        self.define_controller(A, B, C, f, eta, phi)


    @property
    def inertia_matrix(self):
        return self._D0

    @inertia_matrix.setter
    def inertia_matrix(self, matrix:Matrix):
        if self.check_symmetry(matrix):
            if self.check_positive_definite(matrix):
                self._D0 = matrix
            else:
                error_text = "[EulerLagrangeController.inertia_matrix (setter)] The intertia matrix is not positive definite."
                raise ValueError(error_text)
        else:
            error_text = '[EulerLagrangeController.inertia_matrix (setter)] The intertia matrix should be symmetric.'
            raise ValueError(error_text)

    @property
    def damping_matrix(self):
        return self._C0

    @damping_matrix.setter
    def damping_matrix(self, matrix:Matrix):
        if self.check_symmetry(matrix) and matrix.shape[0] == len(self.minimal_states):
            if self.check_positive_definite(matrix):
                self._C0 = matrix
            else:
                error_text = "[EulerLagrangeController.damping_matrix (setter)] The damping matrix is not positive definite."
                raise ValueError(error_text)
        else:
            error_text = '[EulerLagrangeController.damping_matrix (setter)] The damping matrix should be symmetric and should have the same dimension as the states p.'
            raise ValueError(error_text)

    @property
    def stiffness_matrix(self):
        return self._K0

    @stiffness_matrix.setter
    def stiffness_matrix(self, matrix:Matrix):
        if self.check_symmetry(matrix) and matrix.shape[0] == len(self.minimal_states):
            if self.check_positive_definite(matrix):
                self._K0 = matrix
            else:
                error_text = "[EulerLagrangeController.stiffness_matrix (setter)] The stiffness matrix is not positive definite."
                raise ValueError(error_text)
        else:
            error_text = '[EulerLagrangeController.stiffness_matrix (setter)] The stiffness matrix should be symmetric and should have the same dimension as the states p.'
            raise ValueError(error_text)

    @property
    def nonlinear_coefficient_matrix(self):
        return self._C1

    @nonlinear_coefficient_matrix.setter
    def nonlinear_coefficient_matrix(self, matrix:Matrix or None):
        if matrix.shape[0] == matrix.shape[1] and matrix.shape[0] == len(self.minimal_states):
            self._C1 = matrix
        else:
            error_text = '[EulerLagrangeController.stiffness_matrix (setter)] The stiffness matrix should be squared and should have the same dimension as the states p.'
            raise ValueError(error_text)

    @property
    def nonlinear_fcts_callable(self):
        return self._nl_call

    @nonlinear_fcts_callable.setter
    def nonlinear_fcts_callable(self, matrix:Matrix):
        self._nl_call = matrix
    
    @property
    def nonlinear_fcts(self):
        return self._nl

    @nonlinear_fcts.setter
    def nonlinear_fcts(self, matrix:Matrix):
        self.nonlinear_fcts_callable = matrix
        if len(matrix) == len(self.minimal_states):
            Z = zeros(len(self.minimal_states), len(matrix))
            if self.nl_stiffness:
                C = Matrix(BlockMatrix([[self.nonlinear_coefficient_matrix], [Z]]))
            else:
                C = Matrix(BlockMatrix([[Z], [self.nonlinear_coefficient_matrix]]))
            argument = Array(C.T * Matrix(self.states))
            completed_f = []
            for idx, fct in enumerate(matrix):
                if callable(fct[0]):
                    completed_f.append([fct[0](argument[idx])])
                elif fct[0]:
                    completed_f.append(0)
                else:
                    error_text = '[EulerLagrangeController] f should be a callable function or identical 0.'
                    raise AssertionError(error_text)
            self._nl = Matrix(completed_f)
        else:
            error_text = '[EulerLagrangeController.nonlinear_fcts (setter)] The stiffness matrix should have the same row dimension as the number of states.'
            raise ValueError(error_text)

    @property
    def gain_inputs(self):
        return self._NA

    @gain_inputs.setter
    def gain_inputs(self, matrix:Matrix):
        if matrix.shape[1] == len(self.inputs):
            if matrix.shape[0] == len(self.minimal_states):
                self._NA = matrix
            else:
                error_text = '[EulerLagrangeController.gain_inputs (setter)] The input gain matrix should have the same row dimension as the number of states.'
                raise ValueError(error_text)
        else:
            error_text = '[EulerLagrangeController.gain_inputs (setter)] The input gain matrix should have the same column dimension as the number of inputs.'
            raise ValueError(error_text)

    @property
    def gain_dinputs(self):
        return self._NB

    @gain_dinputs.setter
    def gain_dinputs(self, matrix:Matrix):
        if matrix.shape[1] == len(self.dinputs):
            if matrix.shape[0] == len(self.minimal_states):
                self._NB = matrix
            else:
                error_text = '[EulerLagrangeController.gain_dinputs (setter)] The derivate input gain matrix should have the same row dimension as the number of states.'
                raise ValueError(error_text)
        else:
            error_text = '[EulerLagrangeController.gain_dinputs (setter)] The derivate input gain matrix should have the same column dimension as the number of inputs.'
            raise ValueError(error_text)
 

    def create_states(self, size:int, level:int = 0):
        names_list = ['p{}'.format(i + 1) for i in range(size)]
        names_string = ', '.join(names_list)
        if (',' in names_string):
            return Array(dynamicsymbols(names_string, level))
        else:
            return Array([dynamicsymbols(names_string, level)])
            


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

    def check_positive_definite(self, matrix:Matrix):
        eigenv = matrix.eigenvals()
        pos_def_mask = [float(k) > 0 for k, v in eigenv.items()]
        if False in pos_def_mask:
            return False
        else:
            return True

    
    def convert_to_dynamic_controller(self):
        """
        The Euler-Lagrange formalism is transformed to the state and output equation notation of the DynamicController class.

        Returns:
        --------
        result : tuple
            The tuple contains the transformed matrices that are compatible with the function define_controller of DynamicController. 
        """
        dim_states = len(self.minimal_states)
        D0_inv = self.inertia_matrix ** (-1)
        In = eye(dim_states)
        Z = zeros(dim_states)
        K0_D0 = -D0_inv * self.stiffness_matrix
        C0_D0 = -D0_inv * self.damping_matrix
        A = Matrix(BlockMatrix([[Z, In], [K0_D0, C0_D0]]))
        
        C1_D0 = D0_inv * self.nonlinear_coefficient_matrix
        Z = zeros(dim_states, len(self.nonlinear_fcts))
        B = Matrix(BlockMatrix([[Z], [C1_D0]]))

        f = self.nonlinear_fcts

        Z = zeros(dim_states, len(f))
        if self.nl_stiffness:
            C = Matrix(BlockMatrix([[self.nonlinear_coefficient_matrix], [Z]]))
        else:
            C = Matrix(BlockMatrix([[Z], [self.nonlinear_coefficient_matrix]]))
        
        NA_D0 = -D0_inv * self.gain_inputs
        NB_D0 = -D0_inv * self.gain_dinputs
        Z = zeros(dim_states, 1)
        eta = Matrix(BlockMatrix([[Z], [NA_D0 * Matrix(self.minimal_inputs) + NB_D0 * Matrix(self.dinputs)]]))

        phi = self.gain_inputs.T * D0_inv * self.stiffness_matrix ** (-1) * self.inertia_matrix * self.stiffness_matrix * Matrix(self.minimal_states) - self.gain_dinputs.T * D0_inv * self.stiffness_matrix ** (-1) * self.inertia_matrix * self.stiffness_matrix * Matrix(self.minimal_dstates)
        
        return A, B, C, f, eta, phi
