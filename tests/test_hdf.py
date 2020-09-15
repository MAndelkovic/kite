import os
import pytest
import h5py
import kite
import numpy as np

from pybinding.repository import graphene
from .test_utilities import path_from_fixture, temp_file_from_tmp_path, FuzzyEqual
from .conftest import kite_reference_data
from .test_utilities import haldane
from .funcionalities import convert_lattice, convert_hamiltonian, define_type

os.environ['SEED'] = "0"  # sets the seed in KITE random generator
configuration_params = {'length': [128, 128], 'divisions': [2, 2], 'boundaries': [True, True],
                        'is_complex': 1, 'precision': 1, 'spectrum_range': [-4.1, 4.1]}
configuration_calc = {'dos': {'num_points': 1000, 'num_moments': 128, 'num_random': 100, 'num_disorder': 1},
                      'gaussian_wave_packet': {'num_moments': 1024, 'num_points': 1000, 'num_disorder': 2,
                                               'spinor': np.array([[1 / 2 + 1j / 2, 1 / 2 + 1j / 2]]), 'width': 0.5,
                                               'k_vector': np.array([[1 / 2, 1 / 2], [1 / 3, 1 / 3]]),
                                               'mean_value': 0.1,
                                               'timestep': 0.5}
                      }

lattices = {
    'monolayer_graphene': graphene.monolayer(),
    'haldane_model': haldane()
}


@pytest.mark.parametrize("lattice", lattices.values(), ids=lattices.keys())
def test_configuration(lattice):
    configuration = kite.Configuration(**configuration_params)

    spectrum_range = configuration_params['spectrum_range']
    assert FuzzyEqual(configuration.energy_scale, (spectrum_range[1] - spectrum_range[0]) / 2)
    assert FuzzyEqual(configuration.energy_shift, (spectrum_range[1] + spectrum_range[0]) / 2)
    assert FuzzyEqual(configuration.comp, configuration_params['is_complex'])
    assert FuzzyEqual(configuration.prec, configuration_params['precision'])
    assert FuzzyEqual(configuration.bound, configuration_params['boundaries'])
    assert FuzzyEqual(configuration.leng, configuration_params['length'])
    assert FuzzyEqual(configuration.div, configuration_params['divisions'])

    # calculation = kite.Calculation(configuration)
    # req_calc = getattr(calculation, configuration_dos.keys())(**configuration_dos.values())


@pytest.mark.parametrize("lattice", lattices.values(), ids=lattices.keys())
def test_hdf_equality(lattice, kite_reference_data):
    from .funcionalities import key_translate_export

    spectrum_range = configuration_params['spectrum_range']
    energy_scale = (spectrum_range[1] - spectrum_range[0]) / 2
    energy_shift = (spectrum_range[1] + spectrum_range[0]) / 2

    lattice_prop = convert_lattice(lattice=lattice, energy_shift=energy_shift)
    hamiltonian_prop = convert_hamiltonian(num_orbitals=lattice_prop['num_orbitals'],
                                           hoppings=lattice_prop['hoppings'],
                                           space_size=lattice_prop['space_size'],
                                           complx=configuration_params['is_complex'])

    file_ref = kite_reference_data(lattice, configuration_params, configuration_calc, ext='.h5', execute_kite=False)
    with h5py.File(file_ref, 'r') as hf:
        assert hf['IS_COMPLEX'][...] == configuration_params['is_complex']
        assert hf['PRECISION'][...] == configuration_params['precision']
        assert FuzzyEqual(hf['L'][...], configuration_params['length'])
        assert FuzzyEqual(hf['Divisions'][...], configuration_params['divisions'])
        assert hf['DIM'][...] == lattice_prop['space_size']
        assert FuzzyEqual(hf['LattVectors'][...], lattice_prop['latt_vectors'])
        assert FuzzyEqual(hf['OrbPositions'][...], lattice_prop['position'])
        assert hf['NOrbitals'][...] == np.sum(lattice_prop['num_orbitals'])
        assert np.isclose(hf['EnergyScale'][...], energy_scale)
        assert np.isclose(hf['EnergyShift'][...], energy_shift)

        ham_info = hf['Hamiltonian']
        assert FuzzyEqual(ham_info['NHoppings'][...], hamiltonian_prop['num_hoppings'])
        assert FuzzyEqual(ham_info['d'][...], hamiltonian_prop['d'])
        htype = define_type(precision=configuration_params['precision'], is_complex=configuration_params['is_complex'])
        if configuration_params['is_complex']:
            assert FuzzyEqual(ham_info['Hoppings'][...], hamiltonian_prop['t'].astype(htype) / energy_scale)
        else:
            assert FuzzyEqual(ham_info['Hoppings'][...], hamiltonian_prop['t'].real.astype(htype) / energy_scale)

        calc_info = hf['Calculation']
        for keys_calc, values_calc in configuration_calc.items():
            calc = calc_info[keys_calc]
            for keys_ref, values_ref in calc.items():
                if keys_ref in values_calc:
                    try:
                        assert FuzzyEqual(values_ref[...], values_calc.get(key_translate_export[keys_ref]))
                    except:
                        assert FuzzyEqual(values_ref[...][0], values_calc.get(key_translate_export[keys_ref]))
