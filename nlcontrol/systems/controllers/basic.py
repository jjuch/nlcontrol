from nlcontrol.systems.controllers import ControllerBase
from nlcontrol.systems import EulerLagrange

from sympy.tensor.array import Array
from sympy.matrices import eye
from simupy.systems.symbolic import MemorylessSystem

class PID(ControllerBase):
    """
    PID(inputs=w, name="PID")
    PID(ksi0, chi0, psi0, inputs=inputs, name="PID")

    A nonlinear PID controller can be created using the PID class. This class is based on the ControllerBase object. The nonlinear PID is is based on the input vector w(t), containing sympy's dynamicsymbols. The formulation is the following:

    .. math::
        u(t) = \\xi_0(w(t)) + \\chi_0\\left(\\int(w(t),t)\\right) + \\psi_0(w'(t))

    with :math:`.'(t)` indicating the time derivative of the signal. The class object allows the construction of P, PI, PD and PID controllers, by setting chi0 or psi0 to None. The system is based on a MemorylessSystem object from simupy.

    Parameters
    -----------
    args : optional
        ksi0 : array-like
            A list of P-action expressions, containing the input signal.
        chi0 : array-like
            A list of I-action expressions, containing the integral of the input signal.
        psi0 : array-like
            A list of D-action expressions, containing the derivative of the input signal.
    kwargs : 
        inputs : array-like or string
            if `inputs` is a string, it is a comma-separated listing of the input names. If `inputs` is array-like it contains the inputs as sympy's dynamic symbols.
        name : string
            give the system a custom name which will be shown in the block scheme, default: 'PID'.

    Examples
    ---------
    * Create a classic PD controller with two inputs:
        >>> C = PID(inputs='w1, w2')
        >>> w1, w2, w1dot, w2dot = C.create_variables()
        >>> kp = 1, kd = 5
        >>> ksi0 = [kp * w1, kp * w2]
        >>> psi0 = [kd * w1dot, kd * w2dot]
        >>> C.define_PID(ksi0, None, psi0)
    
    * Same as exercise as above, but with a different constructor:
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy import Symbol, diff
        >>> w = dynamicsymbols('w1, w2')
        >>> w1, w2 = tuple(inputs)
        >>> kp = 1, kd = 5
        >>> ksi0 = [kp * w1, kp * w2]
        >>> psi0 = [kd * diff(w1, Symbol('t')), kd * diff(w2, Symbol('t'))]
        >>> C = PID(ksi0, None, psi0, inputs=w)

    * Formulate a standard I-action chi0:
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy import Symbol, integrate
        >>> w = dynamicsymbols('w1, w2')
        >>> w1, w2 = tuple(inputs)
        >>> ki = 0.5
        >>> chi0 = [ki * integrate(w1, Symbol('t')), ki * integrate(w2, Symbol('t'))]
    """

    def __init__(self, *args, **kwargs):
        if 'inputs' not in kwargs.keys():
            error_text = "[nlcontrol.systems.PID] An 'inputs=' keyword is necessary."
            raise AssertionError(error_text)
        else:
            inputs = kwargs['inputs']
        if 'name' not in kwargs.keys():
            kwargs['name'] = "PID"
        states = None
        super().__init__(states, inputs, *args, **kwargs)

        self._ksi0 = None # potential energy shaper
        self._psi0 = None # damping injection
        self._chi0 = None # integral action

        if len(args) not in (0, 1, 3):
            error_text = '[PID] the argument list should contain a P-action vector, or a P-action, I-action, and D-action vector. In the latter case, if I- or D-action is not necessary replace with None.'
            raise AttributeError(error_text)

        if len(args) == 3:
            self.define_PID(*args)
        elif len(args) == 1:
            self.define_PID(*args, None, None)
            

    def __str__(self):
        return """
        PID object:
        ===========
        P: {}
        I: {}
        D: {}
        """.format(self.P_action, self.I_action, self.D_action)


    @property
    def P_action(self):
        return self._ksi0
    
    @P_action.setter
    def P_action(self, fct):
        fct = [fct] if not isinstance(fct, list) and fct is not None else fct
        self._ksi0 = fct

    @property
    def D_action(self):
        return self._psi0

    @D_action.setter
    def D_action(self, fct):
        fct = [fct] if not isinstance(fct, list) and fct is not None else fct
        self._psi0 = fct

    @property
    def I_action(self):
        return self._chi0

    @I_action.setter
    def I_action(self, fct):
        fct = [fct] if not isinstance(fct, list) and fct is not None else fct
        self._chi0 = fct

    def define_PID(self, P, I, D):
        """
        Set all three PID actions with one function, instead of using the setter functions for each individual action. Automatic checking of the dimensions is done as well. The PID's system arguments is set to a simupy's MemorylessSystem object, containing the proper PID expressions. Both P, PI, PD and PID can be formed by setting the appropriate actions to None.

        Parameters
        -----------
        P : list or expression
            A list of expressions or an expression defining ksi0.
        I : list or expression or None
            A list of expressions or an expression defining chi0. If I is None, the controller does not contain an I-action.
        D : list or expression or None
            A list of expressions or an expression defining psi0. If D is None, the controller does not contain a D-action.
        """
        P = [P] if not isinstance(P, list) and fct is not None else P
        dim = len(P)
        self.P_action = P
        if I is None:
            self.I_action = None
        else:
            I = [I] if not isinstance(I, list) and fct is not None else I
            if len(I) == dim:
                self.I_action = I
            else:
                error_text = '[nlcontrol.systems.PID] The dimension of the I vector differs from the dimension of the P vector.'
                raise ValueError(error_text)
        if D is None:
            self.D_action = None
        else:
            D = [D] if not isinstance(D, list) and fct is not None else D
            if len(D) == dim:
                self.D_action = D
            else:
                error_text = '[nlcontrol.systems.PID] The dimension of the D vector differs from the dimension of the P vector.'
                raise ValueError(error_text)
        self.__create_system__()


    def __create_system__(self):
        """
        Create the inputs and output equations from the P, PI,PD, or PID's expressions. 
        """
        if self.I_action is None and self.D_action is None:
            # P-controller
            inputs = self.inputs
            output_equation = Array(self.P_action)
        elif self.I_action is None:
            # PD-controller
            inputs = [val for pair in zip(self.inputs, self.dinputs) for val in pair]
            output_equation = Array([sum(x) for x in zip(self.P_action, self.D_action)])
        elif self.D_action is None:
            # PI-controller
            inputs = [val for pair in zip(self.inputs, self.iinputs) for val in pair]
            output_equation = Array([sum(x) for x in zip(self.P_action, self.I_action)])
        else:
            # PID-controller
            inputs = [val for pair in zip(self.inputs, self.iinputs, self.dinputs) for val in pair]
            output_equation = Array([sum(x) for x in zip(self.P_action, self.I_action, self.D_action)])
        self.inputs = Array(inputs)
        self.system = MemorylessSystem(input_=inputs, output_equation=output_equation)


class EulerLagrangeController(ControllerBase, EulerLagrange):
    """
    EulerLagrangeController(states, inputs, diff_inputs=False, name="EL controller")
    EulerLagrangeController(M, C, K, F, inputs, Qrnc=None, N=None, g=None, diff_inputs=False, name="EL controller")

    A class that defines a controller of the general Euler-Lagrange type. The formalism is given by:

    .. math::
        M(p).p'' + C(p, p').p' + K(p) + Qrnc(p') = F(u, u')

    If Qrnc is specified in the initiator the non-conservative force vector is Qrnc(N*p') with N a vector with constants. If none Nrnc is an identity matrix.

     Here, p represents a minimal state:

    .. math::
        p = [p_1, p_2, ...]^T

    the apostrophe represents a time derivative, :math:`.^T` is the transpose of a matrix, and u is the input vector:

    .. math::
        u = [u_1, u_2, ...]^T
    
    If diff_inputs is True the input vector is

    .. math::
        u^{*} = [u_1, u_2, ..., du_1, du_2, ...]
    
    If diff_inputs is False the input vector is

    .. math::
        u^{*} = u

    A ControllerBase object uses a state equation function of the form:

    .. math::
        p' = h(p, u^{*})

    However, as system contains second time derivatives of the state, an extended state p* is necessary, containing the minimized states and its first time derivatives:

    .. math::
        p^{*} = [p_1, p_2, p_1', p_2', ...]

    which makes it possible to adhere to the ControllerBase formulation:

    .. math::
        p^{*'} = f(p^{*}, u^{*})

    The output is by default the state vector
    
    .. math::
        y = p^{*}

    A custom output equation can be chosen:

    ..math::
        y = g(p^{*}, u^{*})

    Parameters
    -----------
    M : array-like or float
        Inertia matrix, the matrix is positive definite symmetric. Size: n x n
    C : array-like or float
        Coriolis and matrix. Size: m x n
    K : array-like or float
        Stiffness matrix. Size: n x 1
    F : array-like
        Input forces/torque, non-square matrix. Size: n x 1
    inputs : string or array-like
        if `inputs` is a string, it is a comma-separated listing of the input names. If `inputs` is array-like it contains the inputs as sympy's dynamic symbols.
    Qrnc : array-like, optional
        Real non-conservative forces, non-square matrix. If it contains a lambda function element, the argument is replaced by  Size: n x 1
    N : array-like, optional
        non-conservative forces argument is N*p'. Size: n x n
    g : array-like, optional
        Output equation. Size: p x 1
    states : string or array-like
        if `states` is a string, it is a comma-separated listing of the state names. If `states` is array-like it contains the states as sympy's dynamic symbols. If no states argument is given, a default state vector with name 'p' is generated, with size n.
    diff_inputs : boolean, optional
        if true the input vector is expanded with the input derivatives. This also means that the input's dimension is doubled.
    name : string
        give the system a custom name which will be shown in the block scheme, default: 'EL controller'.
    """
    
    def __init__(self, *args, **kwargs):
        if len(args) not in (2, 5):
            error_text = '[EulerLagrangeController] The arguments should be of size 2, specifying the states and the inputs, or size 5, specifying the different matrices of the Euler-Lagrange formalism (see docs).'
            raise AttributeError(error_text)
        
        if len(args) == 2:
            super().__init__(args, kwargs)
        elif len(args) == 5:
            try:
                
                n = 1 if isinstance(args[0], (int, float)) else len(args[0])
            except Exception as e:
                print(e)
                error_text = "[EulerLagrangeController] If five arguments are given, the matrices that are defining the Euler-Lagrange system should be given."
                raise AttributeError(error_text)

            states = 'p0:{}'.format(n)
            inputs = args[4]
            super().__init__(states, inputs=inputs, *kwargs)
            if 'Qrnc' in kwargs.keys():
                Qrnc = kwargs['Qrnc']
                # Implement Nrnc
                if 'Nrnc' in kwargs.keys():
                    Nrnc = kwargs['Nrnc']
                else:
                    Nrnc = eye(n)
                Qrnc_processed = self.__create_term__(Nrnc, Qrnc)
                
            else:
                Qrnc_processed = None
            # TODO: implement F, and g
            self.define_system(args, Qrnc=Qrnc_processed, g=g)
                
    def __create_term__(self, argument, term):
        argument = argument * self.minimal_states
        print(argument)