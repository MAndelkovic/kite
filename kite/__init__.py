try:
    import _kite as _kitecpp
except ImportError as e:
    raise SystemExit('KITE is not compiled properly!')

import os

from .export_kite import *
from .wrapped_func import *


def tests():
    """Run the tests
    """
    import pytest
    import pathlib

    module_path = pathlib.Path(__file__).parent.parent
    previous_dir = os.getcwd()
    os.chdir(os.path.expanduser(str(module_path / 'tests')))

    error_code = pytest.main()

    os.chdir(previous_dir)

    return error_code or None
