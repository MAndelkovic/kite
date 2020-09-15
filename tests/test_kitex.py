import os
import pytest
import h5py
import kite
import numpy as np

from .test_utilities import haldane
from .test_utilities import path_from_fixture, temp_file_from_tmp_path, FuzzyEqual
from .conftest import kite_reference_data
from pybinding.repository import graphene

os.environ['SEED'] = "0"  # sets the seed in KITE random generator
lattices = {
    'monolayer_graphene': graphene.monolayer(),
    'haldane_model': haldane()
}

configuration_params = {'length': [64, 64], 'divisions': [1, 1], 'boundaries': [True, True],
                        'is_complex': True, 'precision': 1, 'spectrum_range': [-9.1, 9.1]}
configuration_calc = {'dos': {'num_points': 100, 'num_moments': 256, 'num_random': 1, 'num_disorder': 1},
                      'conductivity_dc': {'num_points': 50, 'num_moments': 16, 'num_random': 1, 'num_disorder': 1,
                                          'direction': 'xy', 'temperature': 100},
                      'conductivity_optical': {'num_points': 50, 'num_moments': 16, 'num_random': 1,
                                               'num_disorder': 1, 'direction': 'xx', 'temperature': 0.01},
                      }


@pytest.mark.parametrize("lattice", lattices.values(), ids=lattices.keys())
def test_kitex(lattice, kite_reference_data, tmp_path, request):
    from .funcionalities import key_translate_tools, key_translate_tools_output

    file_ref = kite_reference_data(lattice, configuration_params, configuration_calc, ext='.h5')

    with h5py.File(file_ref, 'r') as hf_ref:
        with kite.utilities.block_output():

            configuration = kite.Configuration(**configuration_params)
            calculation = kite.Calculation(configuration)

            for key, val in configuration_calc.items():
                req_calc = getattr(calculation, key)
                req_calc(**val)

            file_name = temp_file_from_tmp_path(request, tmp_path, ext='.h5')
            kite.config_system(lattice, configuration, calculation, filename=file_name)

            kite.KITEx(file_name)
            for desired_calculation, val in configuration_calc.items():
                data_file = str(file_name)[:-3] + '{}.dat'.format(desired_calculation)
                kite.KITEtools('{} --{} -N {} -X'.format(file_name, key_translate_tools[desired_calculation],
                                                         data_file))

            with h5py.File(file_name, 'r') as hf:
                for key in configuration_calc:
                    all_keys = key_translate_tools_output[key]
                    if not isinstance(all_keys, list):
                        all_keys = [all_keys]

                    for key_output in all_keys:
                        assert FuzzyEqual(np.array(hf['Calculation'][key][key_output][...]),
                                          np.array(hf_ref['Calculation'][key][key_output][...]))

                    data_file = str(file_name)[:-3] + '{}.dat'.format(key)

                    func = np.loadtxt(data_file)
                    func_ref = np.loadtxt(str(file_ref)[:-3] + key + '.dat')
                    assert FuzzyEqual(func, func_ref)
