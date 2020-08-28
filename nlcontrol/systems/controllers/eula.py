from sympy.matrices import Matrix, eye, BlockMatrix, zeros
from sympy.tensor.array import Array
from sympy.physics.mechanics import dynamicsymbols

from nlcontrol.systems.controllers import DynamicController

import numpy as np

class EulerLagrangeController(DynamicController):
    def __init__(self, D0, C0, K0, C1, f, NA, NB, inputs):
        self._D0 = None
        self._C0 = None
        self._K0 = None
        self._C1 = None
        self._nl = None
        self._NA = None
        self._NB = None
        self.inputs = inputs
        self.dinputs, _ = self.__create_inputs__()

        if type(D0) in (float, int):
            D0 = [D0]
            C0 = [C0]
            K0 = [K0]
            C1 = [C1]
            f = [f]
        if len(self.inputs) == 1:
            NA = [NA]
            NB = [NB]
        self.inertia_matrix = Matrix(D0)
        self.minimal_states = self.create_states(len(D0))
        self.minimal_dstates = self.create_states(len(D0), level=1)
        self.states = Array(self.minimal_states.tolist() + self.minimal_dstates.tolist())
        self.damping_matrix = Matrix(C0)
        self.stiffness_matrix = Matrix(K0)
        self.nonlinear_stiffness_matrix = Matrix(C1)
        self.nonlinear_stiffness_fcts = f
        self.gain_inputs = Matrix(NA)
        self.gain_dinputs = Matrix(NB)

        # Create system
        super().__init__(states = self.states, inputs = self.inputs)
        A, B, f, eta, phi  = self.convert_to_dynamic_controller()
        self.define_controller(A, B, None, f, eta, phi)


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
    def nonlinear_stiffness_matrix(self):
        return self._C1

    @nonlinear_stiffness_matrix.setter
    def nonlinear_stiffness_matrix(self, matrix:Matrix):
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
            argument = Array(self.nonlinear_stiffness_matrix.T * Matrix(self.minimal_states))
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
        dim_states = len(self.minimal_states)
        D0_inv = self.inertia_matrix ** (-1)
        In = eye(dim_states)
        Z = zeros(dim_states)
        K0_D0 = -D0_inv * self.stiffness_matrix
        C0_D0 = -D0_inv * self.damping_matrix
        A = Matrix(BlockMatrix([[Z, In], [K0_D0, C0_D0]]))
        
        C1_D0 = D0_inv * self.nonlinear_stiffness_matrix
        Z = zeros(dim_states, len(self.nonlinear_stiffness_fcts))
        B = Matrix(BlockMatrix([[Z], [C1_D0]]))

        f = self.nonlinear_stiffness_fcts
        
        NA_D0 = -D0_inv * self.gain_inputs
        NB_D0 = -D0_inv * self.gain_dinputs
        Z = zeros(dim_states, 1)
        eta = Matrix(BlockMatrix([[Z], [NA_D0 * Matrix(self.minimal_inputs) + NB_D0 * Matrix(self.dinputs)]]))

        phi = self.gain_inputs.T * Matrix(self.minimal_states) - self.gain_dinputs.T * Matrix(self.minimal_dstates)
        
        return A, B, f, eta, phi
