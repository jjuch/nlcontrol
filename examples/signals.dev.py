from nlcontrol.signals import step, sinusoid, append, add, impulse, custom

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
sinusoid(amplitude=[1, 2]).simulation(5, plot=False)
sinusoid(amplitude=[0.5, 1.5, 2.5], frequency=[0.1, 0.5, 0.7], phase_shift=[1.5, 2.5, 3.5], y_shift=[0.3, -0.5, 0]).simulation(5, plot=False)

# Impulse
impulse().simulation(5, plot=False)
impulse(3).simulation(5, plot=False)
impulse(2, [0.5, 1]).simulation(5, plot=False)
impulse(amplitude=2).simulation(5, plot=False)
impulse(amplitude=[0.5, 1.5, 2.5], impulse_time=[1.5, 2.5, 3.5]).simulation(5, number_of_samples=10000, plot=False)

# Append
sin = sinusoid(amplitude=2.3, frequency=1.5, phase_shift=-0.3)
step1 = step(step_times=[2.5, 3.5], begin_values=[1.8, 0.5], end_values=[-1.2, 2.1])
step2 = step(step_times=0.2, begin_values=-2.3, end_values=0.8)
append(step1, sin, step2).simulation(5, plot=False)

# Add
sin = sinusoid(2)
step3 = step(step_times=[1.5, 3.5])
add(sin, step3).simulation(5, plot=False)

# Custom and append
t = [0.050,0.100,0.150,0.200,0.250,0.300,0.350,0.400,0.450,0.500,0.550,0.600,0.650,0.700,0.750,0.800,0.850,0.900,0.950]
val_a = [-70.268,-89.339,-90.412,-23.362,5.549,3.897,53.035,81.991,71.318,54.354,42.674,32.902,37.329,54.345,54.461,48.833,38.618,31.380,36.783]
sign1 = custom(t, val_a, interpolation='linear')
val_b = [-67.610,-83.392,-81.538,-36.217,-13.282,-21.712,-21.291,-21.585,-14.860,-3.511,-0.024,-3.338,-1.020,10.848,15.947,17.414,11.081,6.944,9.919]
sign2 = custom(t, val_b)
val_c = [-21.790,-41.517,-34.590,3.681,51.545,74.819,102.913,103.745,68.345,34.296,14.146,2.714,-3.384,-3.794,-0.185,3.421,1.471,-4.369,-3.741]
sign3 = custom(t, val_c)
custom_signal = append(sign1, sign2, sign3)
custom_signal.simulation(1, plot=True)