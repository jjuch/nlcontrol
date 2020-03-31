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
            print('It is not possible to put two systems in serial configuration where the dimension of the output of the first system is not equal to the dimension of the input of the second system.')
            return
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
            print('It is not possible to put two systems in parallel configuration where the dimension of the input of the first system is not equal to the dimension of the input of the second system.')
            return
        elif (self.sys.dim_output != sys_append.sys.dim_output):
            print('It is not possible to put two systems in parallel configuration where the dimension of the output of the first system is not equal to the dimension of the output of the second system.')
            return
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

    
    def simulation(self, initial_conditions, tspan):
        system = self.system
        print(system.output_equation)
        BD = BlockDiagram(self.system)
        system.initial_condition = initial_conditions
        res = BD.simulate(tspan)
        
        x = res.x[:, 0]
        print(res.x)

        plt.figure()
        ObjectLines = plt.plot(res.t, x)
        plt.legend(iter(ObjectLines), [el for el in tuple(self.system.state)])
        plt.title('states versus time')
        plt.xlabel('time (s)')
        plt.show()

        # print(res.y)
        # plt.figure()
        # ObjectLines = plt.plot(res.t, res.y[:,0], res.t, res.y[:,1], res.t, res.y[:, 2])
        # plt.legend(iter(ObjectLines), ['x', 'dx', 'u'])
        # plt.show()
        return res
