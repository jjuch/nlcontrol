from nlcontrol.systems import SystemBase
import nlcontrol.systems.controllers.controller as nlctr_ctr

__all__ = ["toControllerBase"]

def toControllerBase(system: SystemBase):
    if not isinstance(system, SystemBase):
        error_text = "[controller.utils.toControllerBase] The system that needs to be converted should be of the type `SystemBase`."
        raise TypeError(error_text)
    system.name = "controller"
    system.block_type = "controller"
    return system