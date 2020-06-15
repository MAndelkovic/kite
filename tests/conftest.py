import kite
import pytest
import numpy as np
import os.path

from .test_utilities import path_from_fixture
from .funcionalities import key_translate_tools


@pytest.fixture
def kite_reference_data(request):
    """Return baseline data for this `request`. If non exist create it."""

    def get_expected(lattice, config_params, config_calc, group='', ext='', execute_kite=True):
        file = path_from_fixture(request, prefix='reference_data', ext=ext,
                                 override_group=group)

        if file.exists():
            if ext == '.h5':
                return file
            else:
                return np.load(str(file))
        else:
            if not file.parent.exists():
                file.parent.mkdir(parents=True)
            if ext == '.h5':
                with kite.utilities.block_output():
                    configuration = kite.Configuration(**config_params)
                    calculation = kite.Calculation(configuration)

                    for key, val in config_calc.items():
                        req_calc = getattr(calculation, key)
                        req_calc(**val)

                    kite.config_system(lattice, configuration, calculation, filename=file)
                    if execute_kite:
                        kite.KITEx(file)
                        for desired_calculation, val in config_calc.items():
                            kite.KITEtools('{} --{} -N {} -X'.format(file, key_translate_tools[desired_calculation],
                                                                     str(file)[:-3] +
                                                                     '{}.dat'.format(desired_calculation)))

                    return file
            else:
                result = np.zeros(5)
                return np.save(str(file), result)

    return get_expected


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()

    # we only look at actual failing test calls, not setup/teardown
    if rep.when == "call" and rep.failed:
        mode = "a" if os.path.exists("failures") else "w"
        with open("failures", mode) as f:
            # let's also access a fixture for the fun of it
            if "tmpdir" in item.fixturenames:
                extra = " ({})".format(item.funcargs["tmpdir"])
            else:
                extra = ""

            f.write(rep.nodeid + extra + "\n")
