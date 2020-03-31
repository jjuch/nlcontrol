from systems.system import SystemBase
from simupy.systems.symbolic import MemorylessSystem, DynamicalSystem
from sympy.tensor.array import Array
from sympy.functions.special.delta_functions import Heaviside
from sympy import Symbol

states1 = 'x1'
inputs1 = 'u1'
sys1 = SystemBase(states1, inputs1)
x1, x1dot, u1 = sys1.createVariables()
sys1.system = DynamicalSystem(state_equation=Array([-x1 + u1]), state=Array([x1]), output_equation=x1,  input_=u1)


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

mode = 'null'
if mode is 'series':
    series_sys1 = sys1.series(sys2)
    print(series_sys1.sys.state_equation)
    print(series_sys1.sys.output_equation)
    print(series_sys1, ' - ', series_sys1.sys)

    series_sys2 = sys1.series(sys3)
    print(series_sys2.sys.state_equation)
    print(series_sys2.sys.output_equation)
    print(series_sys2, ' - ', series_sys2.sys)

    series_sys3 = sys2.series(sys1)
    print(series_sys3.sys.state_equation)
    print(series_sys3.sys.output_equation)
    print(series_sys3, ' - ', series_sys3.sys)

    series_sys4 = sys2.series(sys2)
    # print(series_sys4.sys.state_equation)
    print(series_sys4.sys.output_equation)
    print(series_sys4, ' - ', series_sys4.sys)
elif mode is 'parallel':
    parallel_sys1 = sys1.parallel(sys2)
    print(parallel_sys1.sys.state_equation)
    print(parallel_sys1.sys.output_equation)
    print(parallel_sys1, ' - ', parallel_sys1.sys)

    parallel_sys2 = sys1.parallel(sys3)
    print(parallel_sys2.sys.state_equation)
    print(parallel_sys2.sys.output_equation)
    print(parallel_sys2, ' - ', parallel_sys2.sys)

    parallel_sys3 = sys2.parallel(sys1)
    print(parallel_sys3.sys.state_equation)
    print(parallel_sys3.sys.output_equation)
    print(parallel_sys3, ' - ', parallel_sys3.sys)

    parallel_sys4 = sys2.parallel(sys2)
    # print(parallel_sys4.sys.state_equation)
    print(parallel_sys4.sys.output_equation)
    print(parallel_sys4, ' - ', parallel_sys4.sys)

input_signal = Heaviside(Symbol('t'))
print(input_signal)
sys1.simulation([1], 20)