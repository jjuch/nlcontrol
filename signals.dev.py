from nlcontrol.signals import step, sinusoid

# Step
step().simulation(5, plot=False)
step(3).simulation(5, plot=False)
step(2, [0.5, 1]).simulation(5, plot=False)
step(step_times=1).simulation(5, plot=False)
step(step_times=[0.5, 1.5, 2.5], begin_values=[0.1, 0.5, 0.7], end_values=[1.5, 2.5, 3.5]).simulation(5, plot=False)

# Sine
sinusoid().simulation(5, plot=False)
sinusoid(3).simulation(5, plot=False)
sinusoid(2, [0.5, 1]).simulation(5, plot=False)
sinusoid(amplitude=1).simulation(5, plot=False)
sinusoid(amplitude=[0.5, 1.5, 2.5], frequency=[0.1, 0.5, 0.7], phase_shift=[1.5, 2.5, 3.5]).simulation(5, plot=False)