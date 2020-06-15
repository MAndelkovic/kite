import os
import sys
from contextlib import contextmanager


@contextmanager
def block_output():
    """ Context manager that temporary blocks the output to std.output """
    with open(os.devnull, 'w') as output:
        init_output = os.dup(sys.stdout.fileno())
        os.dup2(output.fileno(), sys.stdout.fileno())
        try:
            yield
        finally:
            os.dup2(init_output, sys.stdout.fileno())
