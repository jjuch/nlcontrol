from simupy.systems.symbolic import MemorylessSystem
from sympy.physics.mechanics import dynamicsymbols
from sympy.tensor.array import Array

def gain_block(value, dim):
    inputs = dynamicsymbols('x0:{}'.format(dim))
    outputs = Array([value * el for el in inputs])
    return MemorylessSystem(input_=inputs, output_equation=outputs)
