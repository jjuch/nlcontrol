from nlcontrol.signals import step, sinusoid, append, add

# Step
step().simulation(5, plot=False)
step(3).simulation(5, plot=False)
step(2, [0.5, 1]).simulation(5, plot=False)
step(step_times=1).simulation(5, plot=False)
step(step_times=[1, 2]).simulation(5, plot=False)
step(step_times=[0.5, 1.5, 2.5], begin_values=[0.1, 0.5, 0.7], end_values=[1.5, 2.5, 3.5]).simulation(5, plot=False)

# Sine
sinusoid().simulation(5, plot=False)
sinusoid(3).simulation(5, plot=False)
sinusoid(2, [0.5, 1]).simulation(5, plot=False)
sinusoid(amplitude=2).simulation(5, plot=False)
sinusoid(amplitude=[1, 2]).simulation(5, plot=True)
sinusoid(amplitude=[0.5, 1.5, 2.5], frequency=[0.1, 0.5, 0.7], phase_shift=[1.5, 2.5, 3.5], y_shift=[0.3, -0.5, 0]).simulation(5, plot=False)

# Append
sin = sinusoid(amplitude=2.3, frequency=1.5, phase_shift=-0.3)
step1 = step(step_times=[2.5, 3.5], begin_values=[1.8, 0.5], end_values=[-1.2, 2.1])
step2 = step(step_times=0.2, begin_values=-2.3, end_values=0.8)
append(step1, sin, step2).simulation(5, plot=False)

# Add
sin = sinusoid(2)
step3 = step(step_times=[1.5, 3.5])
add(sin, step3).simulation(5, plot=True)