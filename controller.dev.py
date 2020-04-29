from nlcontrol.systems import DynamicController, PID
import numpy as np
from sympy.physics.mechanics import dynamicsymbols
from sympy import Symbol, diff
from sympy.tensor.array import Array


dynsys = DynamicController(inputs='w1', states='z1, z2')
z1, z2, dz1, dz2, w1, dw1 = dynsys.create_variables()

a0 = 12.87
a1 = 6.63
k1 = 0.45
b0 = (48.65 - a1) * k1
b1 = (11.79 - 1) * k1

A = [[0, 1], [-a0, -a1]]
B = [[0], [1]]
C = [[b0], [b1]]

f = lambda t: t


eta = [[w1 + dw1], [(w1 + dw1)**2]]
# eta = [[1],[w1]]
phi = [[z1],  [z2]]

dynsys.define_controller(A, B, C, f, eta, phi)
print(dynsys)

kp = 2.567979797979798
kd = 0.9212822069176614
ksi0 = [kp * w1, 0]
psi0 = [kd * dw1, 0]
pid = PID(ksi0, None, psi0, inputs=Array([w1]))

contr = dynsys.parallel(pid)
print(contr)
