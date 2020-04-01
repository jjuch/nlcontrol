from systems import UMS, Controller
from closedloop.feedback import ClosedLoop

from sympy import cos, sin, eye
import numpy as np

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
# ums.simulate_system([1, 0, np.pi/4, 0], 40, show=False)

states_contr = 'z'
contr = Controller(states_contr, states)
z, z_dot, w1, w2 , w1_dot, w2_dot = contr.createVariables(input_diffs=True)

kp = 0.5
kd = 2
# ksi0 = [kp * w1, kp * w2]
# psi0 = [kd * w1_dot, kd * w2_dot]
ksi0 = [0, kp * w2]
psi0 = [0, kd * w2_dot]
contr.define_linear_part(ksi0, psi0)
print(np.reshape(eye(2), (2,2)))

CL = ClosedLoop(ums.sys, contr.sys)
CL.simulate([1, 0 , np.pi/4, 0], 100)