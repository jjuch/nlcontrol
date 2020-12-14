from nlcontrol.systems.system import TransferFunction
from nlcontrol.signals import step

from control import TransferFunction as TF
from control import tf

num = [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]
den = [[[9., 8., 7.], [6., 5., 4.]], [[3., 2., 1.], [-1., -2., -3.]]]
tf1 = TransferFunction(num, den)
print(tf1)

step_signal = step(step_times=[1, 25], begin_values=[0, 0], end_values=[1, 1])
tf1.simulation(80, input_signals=step_signal, plot=True)

tf2 = TransferFunction(tf1.tf)
print(tf2)

new_tf = tf(num, den)
tf3 = TransferFunction(new_tf)
print(tf3)

