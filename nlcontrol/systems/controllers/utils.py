from nlcontrol.systems import SystemBase
import nlcontrol.systems.controllers.controller as nlctr_ctr

__all__ = ["toControllerBase"]

def toControllerBase(system: SystemBase):
    if not isinstance(system, SystemBase):
        error_text = "[controller.utils.toControllerBase] The system that needs to be converted should be of the type `SystemBase`."
        raise TypeError(error_text)
    controller = nlctr_ctr.ControllerBase(
        inputs=system.inputs,
        states=system.states,
        system=system.system
    )
    if system._additive_output_system is not None:
        controller._additive_output_system = system._additive_output_system
    return controller