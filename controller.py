from eula import EulerLagrange

from sympy.matrices import Matrix
from sympy.tensor.array import Array
from sympy import diff, Symbol
from simupy.systems.symbolic import MemorylessSystem

class Controller(EulerLagrange):
    def __init__(self, states:str, inputs: str):
        super().__init__(states, inputs)
        self._ksi0 = None
        self._psi0 = None
        self._nonLinearPart = False

    @property
    def potential_energy_shaper(self) -> object:
        return self._ksi0
    
    @potential_energy_shaper.setter
    def potential_energy_shaper(self, fct) -> bool:
        self._ksi0 = fct
        return True

    @property
    def damping_injection(self) -> object:
        return self._psi0

    @damping_injection.setter
    def damping_injection(self, fct) -> bool:
        self._psi0 = fct
        return True

    def define_linear_part(self, ksi0, psi0) -> bool:
        # self.potential_energy_shaper = [(-1) * el for el in ksi0]
        # self.damping_injection = [(-1) * el for el in psi0]
        self.potential_energy_shaper = ksi0
        self.damping_injection = psi0
        self.sys = self.create_system()
        return True

    def create_system(self):
        inputs_diff = [diff(input_el, Symbol('t')) for input_el in self.inputs]
        transformed_inputs = [val for pair in zip(self.inputs, inputs_diff) for val in pair]
        if not self._nonLinearPart:
            PD_equation = Array([sum(x) for x in zip(self._ksi0, self._psi0)])
            print(PD_equation)
            return MemorylessSystem(input_=transformed_inputs, output_equation=PD_equation)

    