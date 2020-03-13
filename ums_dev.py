from ums import UMS

from sympy.matrices import Matrix

states = 'q1, q2'
inputs = 'u'

ums = UMS(states, inputs)

print(ums.states[0])

# M = Matrix([[q1,]])