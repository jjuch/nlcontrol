from simupy.systems.symbolic import MemorylessSystem
from sympy.physics.mechanics import dynamicsymbols
from sympy.tensor.array import Array

__all__ = ["gain_block"]

def gain_block(value, dim):
    """
    Multiply the output of system with dimension 'dim' with a contant value 'K'.
    
    .. aafig::
        :aspect: 75
        :scale: 100
        :proportional:
        :textual:
        
        m x 1    +---+  m x 1
        -------->+ K +------->
                 +---+

    Parameters
    -----------
    value : int or float
        Multiply the input signal with a value.
    dim : int

    Returns
    --------
    :obj:`simupy's MemorylessSystem`

    Examples
    ---------
    A negative gain block with dimension 3:
        >>> negative_feedback = gain_block(-1, 3)

    """
    if type(dim) is not int:
        error_text = "[ClosedLoop.blocks] the dimension of the block's input should be an integer."
        raise ValueError(error_text)
    inputs = dynamicsymbols('x0:{}'.format(dim))
    outputs = Array([value * el for el in inputs])
    return MemorylessSystem(input_=inputs, output_equation=outputs)
