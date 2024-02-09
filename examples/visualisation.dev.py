from nlcontrol.systems import EulerLagrange, ControllerBase
from nlcontrol.closedloop import ClosedLoop
from nlcontrol.visualisation import pretty_print_dict

from simupy.systems.symbolic import DynamicalSystem
from sympy.tensor.array import Array

states1 = 'x1, x2'
inputs1 = 'u1, u2'
EL1 = EulerLagrange(states1, inputs1, name="EL1")
x1, x2, dx1, dx2, u1, u2 = EL1.create_variables()
M = [[1, x1*x2],
    [x2*x1, 1]]
C = [[2*dx1, 1 + x1],
    [x2 - 2, 3*dx2]]
K = [x1, 2*x2]
F = [u1 , 0]

EL1.define_system(M, C, K, F)
EL1.show(open_browser=True)

states2 = 'q1, q2'
EL2 = EulerLagrange(states2, EL1.inputs, name="EL2")
q1, q2, dq1, dq2, u1, u2 = EL2.create_variables()
M = [[1, q1*q2],
    [q2*q1, 1]]
C = [[2*dq1, 1 + q1],
    [q2 - 2, 3*dq2]]
K = [q1, 2*q2]
F = [u1 , 0]

EL2.define_system(M, C, K, F)
# EL2.show()

states3 = 'p1, p2'
EL3 = EulerLagrange(states3, EL1.states, name="EL3")
p1, p2, dp1, dp2, u1, u2, u3, u4 = EL3.create_variables()
M = [[1, p1*p2],
    [p2*p1, 1]]
C = [[2*dp1, 1 + p1],
    [p2 - 2, 3*dp2]]
K = [p1, 2*p2]
F = [u1 , 0]

EL3.define_system(M, C, K, F)
# EL3.show()

states_c1 = 'l0:{}'.format(EL1.system.dim_input)
contr1 = ControllerBase(states=states_c1, inputs=EL1.states)
l0, l1, dl0, dl1, u1, u2, u3, u4 = contr1.create_variables()
contr1.system = DynamicalSystem(
    state_equation=Array([-l0 + u1, - l1 + u3]),
    state=Array([l0, l1]),
    output_equation=Array([l0, l1]), 
    input_=Array([u1, u2, u3, u4])
)
# contr1.block_configuration
contr1.show()


######### Test parallel
test_parallel = True
if test_parallel:
    parallel_sys = EL1.parallel(EL2)
    parallel_sys.show(open_browser=True)


    double_parallel = EL1.parallel(parallel_sys)
    double_parallel.show(open_browser=True)
    


    triple_parallel = EL1.parallel(double_parallel)
    triple_parallel.show(open_browser=True)

###### Test series
test_series = True
if test_series:
    series_sys = EL1.series(EL3)
    series_sys.show(open_browser=True)

    double_series = series_sys.series(EL3)
    double_series.show(open_browser=True)

    parallel_series = EL1.parallel(series_sys)
    parallel_series.show(open_browser=True)

    series_parallel = parallel_sys.series(EL3)
    series_parallel.show(open_browser=True)

test_CL = True
if test_CL:
    CL = ClosedLoop(forward=EL1, backward=contr1)
    CL.show()

    double_CL = ClosedLoop(forward=CL.closed_loop_system, backward=contr1)
    double_CL.show()