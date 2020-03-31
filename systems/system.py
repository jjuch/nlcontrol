import sys,traceback

from simupy.block_diagram import BlockDiagram

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
        return SystemBase(self.states.copy(), self.inputs.copy(), sys=self.sys.copy())

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


    def __connect_input(self, input_signal, base_system=None, block_diagram:BlockDiagram=None):
        if base_system is None:
            base_system = self
        elif not isinstance(base_system, SystemBase):
            error_text = '[SystemBase.__connect_input] The system should be an SystemBase instance.'
            raise TypeError(error_text)
        
        if block_diagram is None:
            BD = BlockDiagram(input_signal.system, base_system.system)
        else:
            BD = block_diagram
            BD.add_system(input_signal.system)
            if not base_system.system in set(block_diagram.systems):
                BD.add_system(base_system.system)
        
        BD.connect(input_signal.system, base_system.system)
        return BD

    
    def simulation(self, initial_conditions, tspan, input_signals=None):
        base_system = self.__copy__()

        inputs = input_signals.system
        BD = self.__connect_input(input_signals, base_system=base_system)
        base_system.system.initial_condition = initial_conditions
        res = BD.simulate(tspan)

        plt.figure()
        plt.subplot(121)
        for i in range(base_system.system.dim_state):
            plt.plot(res.t, res.x[:, i], label=base_system.states[i])
        plt.title('states versus time')
        plt.xlabel('time (s)')
        plt.legend()
        plt.subplot(122)
        max_inputs_index = base_system.system.dim_input
        for j in range(max_inputs_index):
            plt.plot(res.t, res.y[:, j], label=base_system.inputs[j])
        for k in range(base_system.system.dim_output):
            plt.plot(res.t, res.y[:, k + max_inputs_index], label='y' + str(k))
        plt.title('inputs and outputs versus time')
        plt.xlabel('time (s)')
        plt.legend()
        plt.show()
        return res
