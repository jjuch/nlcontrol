import nlcontrol.signals as sgnls

import sys,traceback
from copy import deepcopy, copy
from simupy.block_diagram import BlockDiagram, SimulationResult

from sympy.physics.mechanics import dynamicsymbols
from sympy.matrices import Matrix
from sympy.tensor.array.ndim_array import NDimArray
from sympy.physics.mechanics import msubs
from sympy import Symbol, diff
from sympy.tensor.array import Array
from simupy.systems.symbolic import MemorylessSystem, DynamicalSystem

import numpy as np
import matplotlib.pyplot as plt

class SystemBase():
    def __init__(self, states, inputs, sys=None):
        self.states = self._process_init_input(states)
        self.dstates = self._process_init_input(states, 1)
        self.inputs = self._process_init_input(inputs)
        self.sys = sys


    def __copy__(self):
        # states_copy = None if self.states is None else self.states.copy()
        # inputs_copy = None if self.inputs is None else self.inputs.copy()
        # sys_copy = None if self.sys is None else copy(self.sys)
        # return SystemBase(states_copy, inputs_copy, sys_copy)
        return deepcopy(self)

    @property
    def system(self):
        return self.sys

    @system.setter
    def system(self, system):
        self.sys = system  


    def _process_init_input(self, arg:str, level:int=0) -> Matrix:
        '''Return the correct format of the processed __init__input. For a one-element input a different approach to create the parameter is needed.

        Parameters:
            arg [str]: an __init__ input string that needs to be processed. The variables are separated by ','.
            level [int]: Level of differentiation of the returned function.

        Returns:
            matrix [Matrix]: a Matrix of dynamic symbols given by arg. 
        '''
        if arg is None:
            return None
        elif isinstance(arg, NDimArray):
            return arg
        else:
            if (',' in arg):
                return Array(dynamicsymbols(arg, level))
            else:
                return Array([dynamicsymbols(arg, level)])


    def createVariables(self, input_diffs:bool=False) -> tuple:
        '''
        Returns a tuple with all variables. First the states are given, next the derivative of the states, and finally the inputs, optionally followed by the diffs of the inputs.

        Parameters:
            input_diffs [bool]: also return the differentiated versions of the inputs.

        Returns:
            Variables [tuple]: all variables as described above.
        '''
        if self.states is None:
            if len(tuple(self.inputs)) == 1:
                return tuple(self.inputs)[0]
            else:
                return tuple(self.inputs)
        else:
            states_matrix = Matrix(self.states)
            dstates_matrix = Matrix(self.dstates)
            inputs_matrix = Matrix(self.inputs)
            
            var_list_states = dstates_matrix.row_insert(0, states_matrix)
            var_list = inputs_matrix.row_insert(0, var_list_states)
            if input_diffs:
                input_diff_list = Matrix([diff(input_el, Symbol('t')) for input_el in inputs_matrix])
                var_list = input_diff_list.row_insert(0, var_list)
            return tuple(var_list)


    def series(self, sys_append):
        if (self.sys.dim_output != sys_append.sys.dim_input):
            error_text = '[SystemBase.series] Dimension of output of the first system is not equal to dimension of input of the second system.'
            raise ValueError(error_text)
            # raise SystemExit(error_text), None, sys.exc_info()[2]
        else:
            inputs = self.inputs
            substitutions = dict(zip(sys_append.sys.input, self.sys.output_equation))
            output_equations =  Array([msubs(expr, substitutions) for expr in sys_append.sys.output_equation])
            if (self.states is None):
                if (sys_append.states is None):
                    return SystemBase(None, inputs, MemorylessSystem(input_=inputs, output_equation=output_equations))
                else:
                    states = sys_append.states
                    state_equations = Array([msubs(expr, substitutions) for expr in sys_append.sys.state_equation])
                    return SystemBase(states, inputs, DynamicalSystem(state_equation=state_equations, state=states, input_=inputs, output_equation=output_equations))
            else:
                if (sys_append.states is None):
                    states = self.states
                    state_equations = self.sys.state_equation
                else:
                    states = Array(self.states.tolist() + sys_append.states.tolist())
                    state_equations2 = Array(msubs(expr, substitutions) for expr in sys_append.sys.state_equation)
                    state_equations = Array(self.sys.state_equation.tolist() + state_equations2.tolist())
                return SystemBase(states, inputs, DynamicalSystem(state_equation=state_equations, state=states, input_=inputs, output_equation=output_equations))


    def parallel(self, sys_append):
        if (self.sys.dim_input != sys_append.sys.dim_input):
            error_text = '[SystemBase.parallel] Dimension of the input of the first system is not equal to the dimension of the input of the second system.'
            raise ValueError(error_text)
        elif (self.sys.dim_output != sys_append.sys.dim_output):
            error_text = '[SystemBase.parallel] Dimension of the output of the first system is not equal to the dimension of the output of the second system.'
            raise ValueError(error_text)
        else:
            inputs = self.inputs
            substitutions = dict(zip(sys_append.sys.input, self.sys.input))
            output_equations = Array([value[0] + value[1] for value in zip(self.sys.output_equation, [msubs(expr, substitutions) for expr in sys_append.sys.output_equation])])
            if (self.states is None):
                if (sys_append.states is None):
                    return SystemBase(None, inputs, MemorylessSystem(input_=inputs, output_equation=output_equations))
                else:
                    states = sys_append.states
                    state_equations = Array([msubs(expr, substitutions) for expr in sys_append.sys.state_equation])
                    return SystemBase(states, inputs, DynamicalSystem(state_equation=state_equations, state=states, input_=inputs, output_equation=output_equations))
            else:
                if (sys_append.states is None):
                    states = self.states
                    state_equations = self.sys.state_equation
                else:
                    states = Array(self.states.tolist() + sys_append.states.tolist())
                    state_equations2 = Array(msubs(expr, substitutions) for expr in sys_append.sys.state_equation)
                    state_equations = Array(self.sys.state_equation.tolist() + state_equations2.tolist())
                return SystemBase(states, inputs, DynamicalSystem(state_equation=state_equations, state=states, input_=inputs, output_equation=output_equations))


    def __connect_input(self, input_signal=None, base_system=None, block_diagram=None):
        if base_system is None:
            base_system = self
        elif not isinstance(base_system, SystemBase):
            error_text = '[SystemBase.__connect_input] The system should be an SystemBase instance.'
            raise TypeError(error_text)

        if input_signal is None:
            if block_diagram is None:
                BD = BlockDiagram(base_system.system)
            elif not base_system.system in set(block_diagram.systems):
                    BD = block_diagram
                    BD.add_system(base_system.system)
        else:
            if block_diagram is None:
                BD = BlockDiagram(input_signal.system, base_system.system)
            else:
                BD = block_diagram
                BD.add_system(input_signal.system)
                if not base_system.system in set(block_diagram.systems):
                    BD.add_system(base_system.system)
            
            BD.connect(input_signal.system, base_system.system)
        return BD

    
    def simulation(self, tspan, number_of_samples=100, initial_conditions=None, input_signals=None, plot=False):
        base_system = self.__copy__()
        if base_system.states is None:
            if np.isscalar(tspan):
                t = np.linspace(0, tspan, number_of_samples)
            elif len(tspan) == 2:
                t = np.linspace(tspan[0], tspan[1], number_of_samples)
            else:
                t = np.array(tspan)
            self._getOutputEquation()
            y = [base_system.system.output_equation(t_i) for t_i in t]
            if plot:
                plt.figure()
                for k in range(base_system.system.dim_output):
                    plt.plot(t, y[:, k], label='y' + str(k))
                plt.title('inputs and outputs versus time')
                plt.xlabel('time (s)')
                plt.legend()
                plt.show()
            return t, y
            
        else: 
            # if input_signals is None and not (base_system.states is None and base_system.inputs is None):
            #     print('wiwi')
            #     input_signals = sgnls.empty_signal(base_system.system.dim_input, add_states=True)
            BD = self.__connect_input(input_signals, base_system=base_system)
            if initial_conditions is not None:
                if isinstance(initial_conditions, (int, float)):
                    initial_conditions = [initial_conditions]
                base_system.system.initial_condition = initial_conditions

            max_inputs_index = base_system.system.dim_input
            if np.isscalar(tspan):
                res = BD.simulate(tspan)
            elif len(tspan) == 2:
                res = BD.simulate(tspan)
            else:
                res = self.__simulation_loop(tspan, BD, base_system.system)
            
            t = res.t
            x = res.x
            y = res.y[:, max_inputs_index:]
            u = res.y[:, :max_inputs_index]

            if plot:
                plt.figure()
                if base_system.states is not None:
                    plt.subplot(121)
                    for i in range(base_system.system.dim_state):
                        plt.plot(res.t, res.x[:, i], label=base_system.states[i])
                    plt.title('states versus time')
                    plt.xlabel('time (s)')
                    plt.legend()
                    plt.subplot(122)
                for j in range(max_inputs_index):
                    plt.plot(res.t, res.y[:, j], label=base_system.inputs[j])
                for k in range(base_system.system.dim_output):
                    plt.plot(res.t, res.y[:, k + max_inputs_index], label='y' + str(k))
                plt.title('inputs and outputs versus time')
                plt.xlabel('time (s)')
                plt.legend()
                plt.show()
        
            return t, x, y, u, res

    def __simulation_loop(self, time, block_diagram, system_with_states):
        res = SimulationResult(block_diagram.cum_states[-1], block_diagram.cum_outputs[-1],time, block_diagram.systems.size)
        for index, t in enumerate(time):
            if index == 0:
                tspan = [0, t]
            else:
                tspan = [time[index - 1], t]
            res_temp = block_diagram.simulate(tspan)
            # print(index, ': ', tspan, ' - ', res_temp.t)
            res.t[index] = res_temp.t[-1]
            res.x[index] = res_temp.x[-1]
            res.y[index] = res_temp.y[-1]
            res.e[index] = res_temp.e[-1]
            system_with_states.initial_condition = res.x[index]
        return res

    def _getOutputEquation(self):
        print(type(self.system))
        if (isinstance(self.system, DynamicalSystem)):
            print(self.system.output_equation)

    
    def linearize(self, working_point:list):
        """
        A nonlinear system is linearized around a working point.

        Parameters:
        -----------
            working_point : list or int
                the x_dot is linearized around the working point.

        Returns:
        --------

        """
        pass


        