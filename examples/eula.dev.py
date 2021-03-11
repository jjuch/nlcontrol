from nlcontrol.systems import EulerLagrange
from nlcontrol.signals import step
from sympy import atan

states1 = 'x1, x2'
inputs1 = 'u1, u2'
EL1 = EulerLagrange(states1, inputs1)
print(EL1.states)
print(EL1.minimal_states)
print(EL1.create_variables())
x1, x2, dx1, dx2, u1, u2 = EL1.create_variables()
M = [[1, x1*x2],
    [x2*x1, 1]]
C = [[2*dx1, 1 + x1],
    [x2 - 2, 3*dx2]]
K = [x1, 2*x2]
F = [u1 , 0]
Qrnc = [atan(dx1), 0]
g = [x1, x1, x2, x2]

EL1.define_system(M, C, K, F, Qrnc=Qrnc, g=g)
print(EL1)
print("diff(", EL1.states, ") = ", EL1.state_equation)

EL1_lin,_ = EL1.linearize([0, 0, 0, 0], [0, 0])
step_sgnl = step(2)
init_cond = [1, 2, 0.5, 4]
EL1_lin.simulation(5, initial_conditions=init_cond, input_signals=step_sgnl, plot=True)

M1 = EL1.inertia_matrix
C1 = EL1.damping_matrix 
K1 = EL1.stiffness_matrix
F1 = EL1.input_vector
xdot = EL1.state_equation

EL2 = EulerLagrange(EL1.states, EL1.inputs)
print(EL2)
print(EL2.minimal_states)