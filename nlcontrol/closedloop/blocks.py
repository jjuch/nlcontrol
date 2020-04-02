from simupy.systems import LTISystem
from sympy import eye

def gain_block(self, value, dim):
    gain_list = value * eye(dim)
    return LTISystem(gain_list)

