from . import _kitecpp
from . import utilities

def KITEx(input):
    """Calculates the expansion moments for a desired system/functionality defined in the input. For more information
    on available functionalities refer to:
    https://royalsocietypublishing.org/doi/full/10.1098/rsos.191809,
    our github page with examples https://github.com/quantum-kite/kite
    or https://quantum-kite.com/category/capabilities/


    Parameters
    ----------
    input : str, PosixPath
        Name of the h5 file that will be processed with KITEx

    Example
    -------
    The following computes the moments of a function specified in example.h5

        KITEx('example.h5')

    Returns
    -------
    int 0 if the function exited correctly
    """
    try:
        with utilities.block_output():
            if not (isinstance(input, str)):
                return _kitecpp.kitex(str(input))
            else:
                return _kitecpp.kitex(input)
    except ValueError or Exception:
        raise ValueError('Wrong input to KITEx!')


def KITEtools(input):
    """Calculates the reconstructed function from the moments obtained by KITEx and outputs a *.dat file


    Parameters
    ----------
    input : str, PosixPath
        Label that define which function with which parameters will be reconstructed. For a full list of parameters see
        https://quantum-kite.com/category/capabilities/post-processing-tools/

    Example
    -------
    The following reconstruct the DOS taking 1000 moments from example.h5, using Chebyshev-polynomial Green function
    (CPGF) with broadening of 0.01 eV. The computed reconstruction is exported into file dos.dat where the first column
    is an array of energies and the second is the computed DOS

        KITEtools('example.h5 --DOS	-M 1000 -K green 0.01 -N dos.dat')

    Returns
    -------
    int 0 if the function exited correctly
    """

    try:
        with utilities.block_output():
            if not (isinstance(input, str)):
                return _kitecpp.kite_tools(str(input).split())
            else:
                return _kitecpp.kite_tools(input.split())
    except ValueError or Exception:
        raise ValueError('Wrong input to KITEtools!')
