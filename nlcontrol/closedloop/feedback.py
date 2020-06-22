from simupy.block_diagram import BlockDiagram
from sympy.matrices import Matrix

from nlcontrol.closedloop.blocks import gain_block

import numpy as np
import matplotlib.pyplot as plt


class ClosedLoop():
    def __init__(self, system=None, controller=None):
        self.system = system
        self.controller = controller


    def createBlockDiagram(self, forward_systems:list=None, backward_systems:list=None):
        if (forward_systems is None):
            if (self.system is None):
                print('Both the forward_systems argument and the ClosedLoop.system variable are empty. Please provide a forward_system.')
                return
            else:
                forward_systems = [self.system]
        if (backward_systems is None):
            if (self.system is None):
                print('Both the backward_systems argument and the ClosedLoop.controller variable are empty. Please provide a backward_system.')
                return
            else:
                backward_systems = [self.controller]

        BD = BlockDiagram()
        # Order of adding systems is important. The negative feedback_block needs to be added before the backward systems. This can be seen from simupy.block_diagram.BlockDiagram.output_equation_function(). Possibly only for stateless systems important.
        output_startidx_process = 0
        output_endidx_process = -1
        state_startidx_process = 0
        state_endidx_process = -1

        if (len(forward_systems) is not 0):
            for forward_system in forward_systems:
                forward_sys = forward_system.system
                BD.add_system(forward_sys)
                output_endidx_process += forward_sys.dim_output
                state_endidx_process += forward_sys.dim_state
        output_endidx_process += 1
        state_endidx_process += 1

        output_startidx_controller = output_endidx_process
        output_endidx_controller = output_startidx_controller
        state_startidx_controller = state_endidx_process
        state_endidx_controller = state_startidx_controller

        if (len(backward_systems) is not 0):
            negative_feedback = gain_block(-1, backward_systems[-1].system.dim_output)
            BD.add_system(negative_feedback)
            output_startidx_controller += negative_feedback.dim_output
            output_endidx_controller = output_startidx_controller
            for backward_system in backward_systems:
                backward_sys = backward_system.system
                BD.add_system(backward_sys)
                output_endidx_controller += backward_sys.dim_output
                state_endidx_controller += backward_sys.dim_state
        else:
            negative_feedback = gain_block(-1, forward_systems[-1].system.dim_output)
            BD.add_system(negative_feedback)

        for i in range(len(forward_systems)):
            if (i == len(forward_systems) - 1):
                BD.connect(forward_systems[i].system, backward_systems[0].system)
            else:
                BD.connect(forward_systems[i].system, forward_systems[i + 1].system)
        if (len(backward_systems) == 0):
            BD.add_system(negative_feedback)
            BD.connect(forward_systems[-1].system, negative_feedback)
            BD.connect(negative_feedback, forward_systems[0].system)
        else:
            for j in range(len(backward_systems)):
                if (j == len(backward_systems) - 1):
                    BD.connect(backward_systems[j].system, negative_feedback)
                    BD.connect(negative_feedback, forward_systems[0].system)
                else:
                    BD.connect(backward_systems[j].system, backward_systems[j + 1].system)
        
        indices = {
            'process': {
                'output': [output_endidx_process] if output_startidx_process == output_endidx_process\
                    else [output_startidx_process, output_endidx_process],
                'state': None if state_endidx_process is 0\
                    else[state_endidx_process] if state_startidx_process == state_endidx_process\
                    else [state_startidx_process, state_endidx_process]
            },
            'controller': {
                'output': [output_endidx_controller] if output_startidx_controller == output_endidx_controller\
                    else [output_startidx_controller, output_endidx_controller],
                'state': None if state_endidx_controller == state_endidx_process\
                    else[state_endidx_controller] if state_startidx_controller == state_endidx_controller\
                    else [state_startidx_controller, state_endidx_controller]
            }
        }
        return BD, indices


    def simulation(self, initial_conditions, tspan, custom_integrator_options=None, plot=False):
        if custom_integrator_options is not None:
            integrator_options = {
                'name': custom_integrator_options['name'] if 'name' in custom_integrator_options else 'dopri5',
                'rtol': custom_integrator_options['rtol'] if 'rtol' in custom_integrator_options else 1e-6,
                'atol': custom_integrator_options['atol'] if 'atol' in custom_integrator_options else 1e-12,
                'nsteps': custom_integrator_options['nsteps'] if 'nsteps' in custom_integrator_options else 500,
                'max_step': custom_integrator_options['max_step'] if 'max_step' in custom_integrator_options else 0.0
            }
        else:
            integrator_options = {
                'name': 'dopri5',
                'rtol': 1e-6,
                'atol': 1e-12,
                'nsteps': 500,
                'max_step': 0.0
            }

        BD, indices = self.createBlockDiagram()
        self.system.system.initial_condition = initial_conditions
        res = BD.simulate(tspan, integrator_options=integrator_options)
        
        # Unpack indices
        y_p_idx = indices['process']['output']
        x_p_idx = indices['process']['state']
        y_c_idx = indices['controller']['output']
        x_c_idx = indices['controller']['state']
        y_p = res.y[:, y_p_idx[0]] if len(y_p_idx) == 1\
            else res.y[:, slice(*y_p_idx)]
        x_p = None if x_p_idx is None\
            else res.x[:, x_p_idx[0]] if len(x_p_idx) == 1\
            else res.x[:, slice(*x_p_idx)]
        y_c = res.y[:, y_c_idx[0]] if len(y_c_idx) == 1\
            else res.y[:, slice(*y_c_idx)]
        x_c = None if x_c_idx is None\
            else res.x[:, x_c_idx[0]] if len(x_c_idx) == 1\
            else res.x[:, slice(*x_c_idx)]

    
        if plot:
            plt.figure()
            plt.subplot(1, 2, 1)
            ObjectLines1A = plt.plot(res.t, y_p)
            ObjectLines1B = plt.plot(res.t, y_c)
            plt.legend(iter(ObjectLines1A + ObjectLines1B), ['y' + str(index) for index in range(1, len(y_p[0]) + 1)] + ['u' + str(index) for index in range(1, len(y_c[0]) + 1)])
            plt.title('Outputs')
            plt.xlabel('time [s]')

            plt.subplot(1, 2, 2)
            ObjectLines2A = []
            ObjectLines2B = []
            if x_p is not None:
                ObjectLines2A = plt.plot(res.t, x_p)
            if x_c is not None:
                ObjectLines2B = plt.plot(res.t, x_c)
            plt.legend(iter(ObjectLines2A + ObjectLines2B), ['x' + str(index) for index in ([] if x_p is None else range(1, len(x_p[0]) + 1))] + ['z' + str(index) for index in ([] if x_c is None else range(1, len(x_c[0]) + 1))])
            plt.title('States')
            plt.xlabel('time [s]')
            plt.show()

        return res.t, (x_p, y_p, x_c, y_c)
