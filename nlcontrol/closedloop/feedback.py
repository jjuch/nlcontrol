from simupy.block_diagram import BlockDiagram
from sympy.matrices import Matrix

from nlcontrol.closedloop.blocks import gain_block

import numpy as np
import matplotlib.pyplot as plt


class ClosedLoop():
    def __init__(self, system:None, controller:None):
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
        if (len(forward_systems) is not 0):
            for forward_system in forward_systems:
                BD.add_system(forward_system)
        if (len(backward_systems) is not 0):
            negative_feedback = gain_block(-1, backward_systems[-1].dim_output)
            BD.add_system(negative_feedback)
            for backward_system in backward_systems:
                print('backward_system: ', backward_system.dim_output)
                BD.add_system(backward_system)
        else:
            negative_feedback = gain_block(-1, forward_systems[-1].dim_output)
            BD.add_system(negative_feedback)
        for i in range(len(forward_systems)):
            if (i == len(forward_systems) - 1):
                BD.connect(forward_systems[i], backward_systems[0])
            else:
                BD.connect(forward_systems[i], forward_systems[i + 1])
        if (len(backward_systems) == 0):
            BD.add_system(negative_feedback)
            BD.connect(forward_systems[-1], negative_feedback)
            BD.connect(negative_feedback, forward_systems[0])
        else:
            for j in range(len(backward_systems)):
                if (j == len(backward_systems) - 1):
                    BD.connect(backward_systems[j], negative_feedback)
                    BD.connect(negative_feedback, forward_systems[0])
                    # BD.connect(backward_systems[j], forward_systems[0])
                else:
                    BD.connect(backward_systems[j], backward_systems[j + 1])
        return BD


    def simulate(self, initial_conditions, tspan):
        # negative_feedback = gain_block(-1, 2)
        # BD = BlockDiagram(self.system, negative_feedback, self.controller)
        # BD.connect(self.system, self.controller)
        # BD.connect(self.controller, negative_feedback)
        # BD.connect(negative_feedback, self.system)
        BD = self.createBlockDiagram()
        self.system.initial_condition = initial_conditions
        res = BD.simulate(tspan)
        # attrs = vars(res)
        # print(', '.join("%s: %s" % item for item in attrs.items()))
        x = res.x[:, 0]
        theta = res.x[:, 2]

        plt.figure()
        ObjectLines = plt.plot(res.t, x, res.t, theta)
        plt.legend(iter(ObjectLines), [el for el in tuple(self.system.state)])
        plt.title('states versus time')
        plt.xlabel('time (s)')
        plt.show()

        # error_sign = [-theta - 5*dtheta for theta, dtheta in zip(res.x[:, 2], res.x[:,3])]
        plt.figure()
        ObjectLines = plt.plot(res.t, res.y[:,0],res.t, res.y[:,1],res.t, res.y[:,2],res.t, res.y[:,3],res.t, res.y[:,4],res.t, res.y[:,5],res.t, res.y[:,6],res.t, res.y[:,7])
        plt.legend(iter(ObjectLines), ['x', 'dx', 'theta', 'dtheta', 'u1', 'u2', 'u3', 'u4'])
        plt.show()