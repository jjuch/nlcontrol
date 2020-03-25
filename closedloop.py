from simupy.block_diagram import BlockDiagram

import numpy as np
import matplotlib.pyplot as plt


class ClosedLoop():
    def __init__(self, system, controller):
        self.system = system
        self.controller = controller

    def createBlockDiagram(self):
        BD = BlockDiagram(self.system, self.controller)
        BD.connect(self.system, self.controller)
        BD.connect(self.controller, self.system)
        return BD

    def simulate(self, initial_conditions, tspan):
        BD = BlockDiagram(self.system, self.controller)
        BD.connect(self.system, self.controller)
        BD.connect(self.controller, self.system)
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

        plt.figure()
        ObjectLines = plt.plot(res.t, res.y[:,0],res.t, res.y[:,1],res.t, res.y[:,2],res.t, res.y[:,3],res.t, res.y[:,4],res.t, res.y[:,5])
        plt.legend(iter(ObjectLines), ['x', 'dx', 'theta', 'dtheta', 'u', 'du'])
        plt.show()