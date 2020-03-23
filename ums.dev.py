from ums import UMS 
from sympy import cos, sin
import numpy as np

states = 'x, theta'
inputs = 'L1, L2'

ums = UMS(states, inputs)

x, theta, dx, dtheta, L1, L2 = ums.createVariables()

e = 0.5
M = [[1, e*cos(theta)], [e*cos(theta), 1]]
C = [[0, (-1)*e*dtheta*sin(theta)], [0, 0]]
K = [[x],[0]]
F = [[0], [L2]]

ums.define_system(M, C, K, F)
ums.simulate_system([1, 0, np.pi/4, 0], 40, show=True)

print(ums._M)
print(ums.inertia_matrix)