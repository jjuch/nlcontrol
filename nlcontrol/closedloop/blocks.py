from simupy.systems.symbolic import MemorylessSystem
from sympy.physics.mechanics import dynamicsymbols
from sympy.tensor.array import Array

__all__ = ["gain_block", "summation_block"]

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
        error_text = "[ClosedLoop.blocks] the dimension of the gain block's input should be an integer."
        raise ValueError(error_text)
    inputs = dynamicsymbols('inp_g0:{}'.format(dim))
    outputs = Array([value * el for el in inputs])
    return MemorylessSystem(input_=inputs, output_equation=outputs)


def summation_block(dim):
    """
    Add the output of system1 and system2 together. Both systems have the same output dimension 'dim'.
    
    .. aafig::
        :aspect: 75
        :scale: 100
        :proportional:
        :textual:

        m x 1   +----+
        ------->+    | m x 1
                |  + |----->
        ------->+    |
        m x 1   +----+
        
    Parameters
    -----------
    dim : int
        The input dimension. The block will have input dimension 2 * m.

    Returns
    --------
    :obj:`simupy's MemorylessSystem`

    Examples
    ---------
    A summation block with dimension 3:
        >>> sum_block = summation_block(3)

    """
    if type(dim) is not int:
        error_text = "[ClosedLoop.blocks] the dimension of the summation block's input should be an integer."
        raise ValueError(error_text)
    dim_in = 2 * dim
    inputs = dynamicsymbols('inp_s0:{}'.format(dim_in))
    outputs = Array([el1 + el2 for el1, el2 in zip(inputs[0: dim], inputs[dim:])])
    return MemorylessSystem(input_=inputs, output_equation=outputs)
    
if __name__ == "__main__":
    sum_block = summation_block(3)
    print(sum_block.input)
    print(sum_block.output_equation)