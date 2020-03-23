from eula import EulerLagrange

class Controller(EulerLagrange):
    def __init__(self, states:str, inputs: str, A, B):
        super().__init__(states, inputs)
        self._A = A
        self._B = B

    def test(self):
        print(self._A)
        print(self.dstates)