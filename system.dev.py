from system import SystemBase
from simupy.systems.symbolic import MemorylessSystem, DynamicalSystem
from sympy.tensor.array import Array

states1 = 'x1'
inputs1 = 'u1'
sys1 = SystemBase(states1, inputs1)
x1, x1dot, u1 = sys1.createVariables()
sys1.system = DynamicalSystem(state_equation=Array([x1 + u1]), state=Array([x1]), output_equation=x1,  input_=u1)


states2 = None
inputs2 = 'w'
sys2 = SystemBase(states2, inputs2)
w = sys2.createVariables()
sys2.sys = MemorylessSystem(input_=Array([w]), output_equation= Array([5 * w]))

states3 = 'x2'
inputs3 = 'u2'
sys3 = SystemBase(states3, inputs3)
x2, x2dot, u2, u2dot = sys3.createVariables(True)
sys3.system = DynamicalSystem(state_equation=Array([x2 - u2**2]), state=Array([x2]), output_equation=Array([x2]),  input_=u2)

test_sys1 = sys1.series(sys2)
print(test_sys1.sys.state_equation)
print(test_sys1.sys.output_equation)
print(test_sys1, ' - ', test_sys1.sys)

test_sys2 = sys1.series(sys3)
print(test_sys2.sys.state_equation)
print(test_sys2.sys.output_equation)
print(test_sys2, ' - ', test_sys2.sys)

test_sys3 = sys2.series(sys1)
print(test_sys3.sys.state_equation)
print(test_sys3.sys.output_equation)
print(test_sys3, ' - ', test_sys3.sys)

test_sys4 = sys2.series(sys2)
# print(test_sys4.sys.state_equation)
print(test_sys4.sys.output_equation)
print(test_sys4, ' - ', test_sys4.sys)

