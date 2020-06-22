from nlcontrol.systems import EulerLagrange, PID, SystemBase
from nlcontrol.closedloop.feedback import ClosedLoop
from nlcontrol.closedloop.blocks import gain_block
from nlcontrol.signals import step

from sympy import cos, sin, eye
import numpy as np
from simupy.block_diagram import BlockDiagram

states = 'x, theta'
inputs = 'L1, L2'

ums = EulerLagrange(states, inputs)

x, theta, dx, dtheta, L1, L2 = ums.create_variables()

e = 0.5
M = [[1, e*cos(theta)], [e*cos(theta), 1]]
C = [[0, (-1)*e*dtheta*sin(theta)], [0, 0]]
K = [[x],[0]]
F = [[L1], [L2]]

ums.define_system(M, C, K, F)
# ums.simulation(40, initial_conditions=[1, 0, np.pi/4, 0], plot=True)

kp = 1
kd = 5
ksi0 = [0, kp * theta]
psi0 = [0, kd * dtheta]
contr = PID(ksi0, None, psi0, inputs=ums.minimal_states)
print(contr)


CL = ClosedLoop(ums, contr)
CL.simulation(5, [1, 0 , np.pi/4, 0], 100)