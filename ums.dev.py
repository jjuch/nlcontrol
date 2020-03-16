from ums import UMS
from sympy import symbols, cos, sin

states = 'x, theta'
inputs = 'L1, L2'

ums = UMS(states, inputs)

x, theta, dx, dtheta, L1, L2 = ums.createVariables()

e = 0.5
M = [[1, e*cos(theta)], [e*cos(theta), 1]]
C = [[0, (-1)*e*dtheta*sin(theta)], [0, 0]]
K = [[x],[0]]
F = [[L1], [L2]]

ums.define_system(M, C, K, F)

print(ums._M)
print(ums.inertia_matrix)