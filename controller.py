from eula import EulerLagrange

class Controller(EulerLagrange):
    def __init__(self, states, inputs):
        super().__init__(states, inputs)