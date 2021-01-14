from nlcontrol.systems import SystemBase
from nlcontrol.signals import step, empty_signal
from simupy.block_diagram import BlockDiagram
from simupy.systems.symbolic import MemorylessSystem, DynamicalSystem
from simupy.systems import SystemFromCallable
from sympy.tensor.array import Array
from sympy import Symbol, sin

import numpy as np

# states1 = 'x1'
# inputs1 = 'u1'
# sys1 = SystemBase(states1, inputs1)
# x1, x1dot, u1 = sys1.create_variables()
# # Two ways possible
# # option 1
# sys1.system = DynamicalSystem(state_equation=Array([-x1 + u1]), state=x1, output_equation=x1,  input_=u1)
# # option 2
# sys1.set_dynamics(output_equation=x1, state_equation=[-x1 + u1])
# sys1_lin, _ = sys1.linearize(1)
# print('state_eq: ', sys1_lin.system.state_equation)

states1a = 'x1a'
inputs1a = 'u1a'
sys1a = SystemBase(states1a, inputs1a)
x1a, x1adot, u1a = sys1a.create_variables()
sys1a.set_dynamics(output_equation=[x1a + u1a + sin(u1a), x1a - u1a + 1, 0], state_equation=[-x1a + u1a])
print(sys1a)
sys1a_lin, _ = sys1a.linearize(1)
print(sys1a_lin)
print('output_eq: ', sys1a_lin.output_equation)
exit()


# states2 = None
# inputs2 = 'w'
# sys2 = SystemBase(states2, inputs2)
# w = sys2.create_variables()
# output_eq = Array([5 * w])
# # Two ways possible
# # option 1
# # sys2.sys = MemorylessSystem(input_=Array([w]), output_equation=output_eq)
# # option 2
# sys2.set_dynamics(output_equation=[5*w])
# print('output_eq: ', sys2.output_equation)


# states3 = 'x2'
# inputs3 = 'u2'
# sys3 = SystemBase(states3, inputs3)
# x2, x2dot, u2, u2dot = sys3.create_variables(True)
# # sys3.system = DynamicalSystem(state_equation=Array([-x2**2 - u2**2]), state=Array([x2]), output_equation=Array([x2]),  input_=u2)
# sys3.set_dynamics(output_equation=[x2], state_equation=[-x2**2 - u2**2])
# sys3_lin, _ = sys3.linearize(1, 2)
# print('state_eq: ', sys3_lin.system.state_equation)


states4 = 'x3, x4'
inputs4 = 'u3'
sys4 = SystemBase(states4, inputs4)
print('Vars: ', sys4.create_variables())
x3, x4, x3dot, x4dot, u3 = sys4.create_variables()
sys4.system = DynamicalSystem(state_equation=Array([-x3 + x4**2 + u3, -x4 + 0.5 * x3]), state=Array([x3, x4]), output_equation=Array([x3 * x4, x4]), input_=u3)
sys4.set_dynamics(output_equation=[x3 * x4, x4], state_equation=[-x3 + x4**2 + u3, -x4 + 0.5 * x3])
sys4_lin, _ = sys4.linearize([2, 1], 1)
print('state_eq: ', sys4_lin.system.state_equation)
exit()

states5 = 'x5'
inputs5 = 'u4, u5'
sys5 = SystemBase(states5, inputs5)
x5, x5dot, u4, u5 = sys5.create_variables()
sys5.system = DynamicalSystem(state_equation=Array([-x5 + u4 - u5]), state=Array([x5]), output_equation=Array([x5]), input_=Array([u4, u5]))


mode = 'series'
if mode is 'series':
    series_sys1 = sys1.series(sys2)
    print(series_sys1.sys.state_equation)
    print(series_sys1.sys.output_equation)
    print(series_sys1, ' - ', series_sys1.sys)

    series_sys2 = sys1.series(sys3)
    print(series_sys2.sys.state_equation)
    print(series_sys2.sys.output_equation)
    print(series_sys2, ' - ', series_sys2.sys)

    sys2.block_configuration
    sys1.block_configuration
    series_sys3 = sys2.series(sys1)
    print(series_sys3.sys.state_equation)
    print(series_sys3.sys.output_equation)
    print(series_sys3, ' - ', series_sys3.sys)

    series_sys4 = sys2.series(sys2)
    # print(series_sys4.sys.state_equation)
    print(series_sys4.sys.output_equation)
    print(series_sys4, ' - ', series_sys4.sys)

    series_md = sys4.series(sys5)
    print(series_md.sys.state)
    print(series_md.sys.state_equation)
    print(series_md.sys.output_equation)
    print(series_md, ' - ', series_md.sys)
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

input_step1 = step()
input_step2 = step(step_times=[5, 15], end_values=[0.9, 1.1], begin_values=[0.2, 0.15])
input_step3 = step(step_times=[5], end_values=[1.4], begin_values=[0.4])
input_step4 = step(step_times=[5, 5], end_values=[1.4, 1.5], begin_values=[0.4, 0.5])
input_empty = empty_signal(sys2.system.dim_input)


time_axis = np.linspace(0, 20, 100)

test_simulation = False
if test_simulation:
    # 1
    sys1.simulation(time_axis, initial_conditions=1, input_signals=input_step3, plot=True)
    # 2
    series_md.simulation(20, initial_conditions=[0.1, 0.5, 0.2], input_signals=input_step3, plot=True)
    # 3
    series_md.simulation(20, initial_conditions=[0.1, 0.5, 0.2], plot=True)
    # 4A
    sys5.simulation([2, 20], initial_conditions=[0.5], input_signals=input_step2, plot=True)
    # 4B
    sys5.simulation([2, 20], initial_conditions=[0.5], input_signals=input_step4, plot=True)
    # 5
    sys2.simulation(time_axis, plot=True, input_signals=input_step3)
    #6
    sys2.simulation([0, 15], initial_conditions=1, plot=True)
    # 7
    input_step2.simulation(time_axis, plot=True)

sys5.simulation([2, 20], initial_conditions=[0.5], input_signals=input_step2, plot=False)
integrator_options = {'nsteps': 1000}
sys5.simulation([2, 20], initial_conditions=[0.5], input_signals=input_step2, plot=True, custom_integrator_options=integrator_options)