from sympy.matrices import Matrix, eye, BlockMatrix, zeros
from sympy.tensor.array import Array
from sympy.physics.mechanics import dynamicsymbols

from nlcontrol.systems.controllers import DynamicController

import numpy as np

class EulerLagrangeController(DynamicController):
    """
    EulerLagrangeController(D0, C0, K0, C1, f, NA, NB, inputs, nonlinearity_type='stiffness')

    The EulerLagrangeController object is based on the DynamicController class. The control equation is:
        D0.p'' + C0.p' + K0.p + C1.f(C1^T.p) + N0.w' = 0
    The apostrophe represents a time derivative, ^T is the transpose of the matrix. 

    The output equation is:
        NA^T.D0^(-1).K0^(-1).D0.K0.p - NB^T.D0^(-1).K0^(-1).D0.K0.p'
    
    More info on the controller can be found in [1, 2]. Here, the parameter \bar{gamma} = 0 and \bar{alpha} = I, with I the identity matrix.

    Parameters:
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

    Attributes:
    -----------
        D0 : inertia_matrix
            Inertia forces.
        C0 : damping_matrix
            Damping and Coriolis forces.
        K0 : stiffness_matrix
            Elastic en centrifugal forces.
        C1 : nonlinear_coefficient_matrix
            Coëfficient of the nonlinear functions.
        nl : nonlinear_stiffness_fcts
            Nonlinear functions of the controller.
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
        
        
    Examples:
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
    ----------
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
        self.nonlinear_stiffness_fcts = f
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
    def nonlinear_stiffness_fcts(self):
        return self._nl

    @nonlinear_stiffness_fcts.setter
    def nonlinear_stiffness_fcts(self, matrix:Matrix):
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
            error_text = '[EulerLagrangeController.nonlinear_stiffness_fcts (setter)] The stiffness matrix should have the same row dimension as the number of states.'
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
        Z = zeros(dim_states, len(self.nonlinear_stiffness_fcts))
        B = Matrix(BlockMatrix([[Z], [C1_D0]]))

        f = self.nonlinear_stiffness_fcts

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
