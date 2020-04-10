from nlcontrol.systems import EulerLagrange

states1 = 'x1, x2'
inputs1 = 'u1'
EL1 = EulerLagrange(states1, inputs1)
print(EL1.states)
print(EL1.minimal_states)
print(EL1.create_variables())
x1, x2, dx1, dx2, u1 = EL1.create_variables()
M = [[1, x1*x2],
    [x2*x1, 1]]
C = [[2*dx1, 1 + x1],
    [x2 - 2, 3*dx2]]
K = [1, 2]
F = [u1 , 0]

EL1.define_system(M, C, K, F)
print("diff(", EL1.states, ") = ", EL1.state_equation)


EL2 = EulerLagrange(EL1.states, EL1.inputs)
print(EL2.states)
print(EL2.minimal_states)