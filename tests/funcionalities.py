import numpy as np

key_translate_tools = {'dos': 'DOS',
                       'conductivity_dc': 'CondDC',
                       'conductivity_optical': 'CondOpt',
                       'conductivity_optical_nonlinear': 'CondOpt2',
                       'ldos': 'LDOS', 'arpes': 'ARPES'}

key_translate_tools_output = {'dos': 'MU',
                              'conductivity_dc': 'Gammaxy',
                              'conductivity_optical': ['Gammaxx', 'Lambdaxx'],

                              }
key_translate_export = {'NumMoments': 'num_moments',
                        'NumRandoms': 'num_random',
                        'NumPoints': 'num_points',
                        'NumDisorder': 'num_disorder', 'k_vector': 'k_vector', 'width': 'width', 'spinor': 'spinor',
                        'timestep': 'timestep',
                        'probing_point': 'probing_point', 'mean_value': 'mean_value'}


def convert_hamiltonian(num_orbitals, hoppings, space_size, complx):
    """ Convert the Hamiltonian from PB to KITE format """
    from scipy.sparse import coo_matrix
    orbital_from = []
    orbital_to = []
    orbital_hop = []
    # number of orbitals before i-th sublattice, where is is the array index
    orbitals_before = np.cumsum(num_orbitals) - num_orbitals
    # iterate through all hoppings, and define unique orbital hoppings
    # orbital_to in unit cell [i, j] is defined  as [i, j] x [1, 3] + relative_orbital_num*3**2 2D
    # orbital_to in unit cell [i, j, k] is defined  as [i, j, k] x [1, 3, 9] + relative_orbital_num*3**3 3D
    # relative index of orbital_from is unique as only hoppings from the orbitals in the initial unit cell are exported
    for h in hoppings:
        hopping_energy = h['hopping_energy']
        it = np.nditer(hopping_energy, flags=['multi_index'])
        while not it.finished:
            relative_move = np.dot(h['relative_index'] + 1,
                                   3 ** np.linspace(0, space_size - 1, space_size, dtype=np.int32))
            # if hopping_energy.size > 1:
            orbital_from.append(orbitals_before[h['from_id']] + it.multi_index[0])
            orbital_to.append(relative_move + (orbitals_before[h['to_id']] + it.multi_index[1]) * 3 ** space_size)
            # else:
            #     orbital_from.append(h['from_id'])
            #     orbital_to.append(relative_move + h['to_id'] * 3 ** space_size)

            orbital_hop.append(it[0] if complx else np.real(it[0]))

            it.iternext()

    # extract t - hoppings where each row corresponds to hopping from row number orbital and d - for each hopping it's
    # unique identifier
    t_list = []
    d_list = []
    # make a sparse matrix from orbital_hop, and (orbital_from, orbital_to) as it's easier to take nonzero hoppings from
    # sparse matrix
    matrix = coo_matrix((orbital_hop, (orbital_from, orbital_to)),
                        shape=(np.max(orbital_from) + 1, np.max(orbital_to) + 1))
    # num_hoppings is a vector where each value corresponds to num of hoppings from orbital equal to it's index
    num_hoppings = np.zeros(matrix.shape[0])
    # iterate through all rows of matrix, number of row = number of orbital from

    for i in range(matrix.shape[0]):
        # all hoppings from orbital i
        row_mat = matrix.getrow(i)
        # number of hoppings from orbital i
        num_hoppings[i] = row_mat.size

        t_list.append(row_mat.data)
        d_list.append(row_mat.indices)

    # fix the size of hopping and distance matrices, where the number of columns is max number of hoppings
    max_hop = int(np.max(num_hoppings))
    d = np.zeros((matrix.shape[0], max_hop))
    t = np.zeros((matrix.shape[0], max_hop), dtype=matrix.data.dtype)
    for i_row, d_row in enumerate(d_list):
        t_row = t_list[i_row]
        d[i_row, :len(d_row)] = d_row
        t[i_row, :len(t_row)] = t_row

    return {'num_hoppings': num_hoppings, 't': t, 'd': d}


def convert_lattice(lattice, energy_shift):
    vectors = np.asarray(lattice.vectors)
    space_size = vectors.shape[0]
    vectors = vectors[:, 0:space_size]

    position_atoms = np.zeros([lattice.nsub, space_size], dtype=np.float64)
    num_orbitals = np.zeros(lattice.nsub, dtype=np.int64)
    hoppings = []
    for name, sub in lattice.sublattices.items():
        # num of orbitals at each sublattice is equal to size of onsite energy
        num_energies = np.asarray(sub.energy).shape[0]
        num_orbitals[sub.alias_id] = num_energies
        # position_atoms is a list of vectors of size space_size
        position_atoms[sub.alias_id, :] = sub.position[0:space_size]
        # define hopping dict from relative hopping index from and to id (relative number of sublattice in relative
        # index lattice) and onsite
        # energy shift is substracted from onsite potential, this is later added to the hopping dictionary,
        # hopping terms shouldn't be substracted
        hopping = {'relative_index': np.zeros(space_size, dtype=np.int32), 'from_id': sub.alias_id,
                   'to_id': sub.alias_id, 'hopping_energy': sub.energy - energy_shift}
        hoppings.append(hopping)

    position_atoms = np.array(position_atoms)
    position = np.repeat(position_atoms, num_orbitals, axis=0)
    # iterate through all the hoppings and add hopping energies to hoppings list
    for name, hop in lattice.hoppings.items():
        hopping_energy = hop.energy
        for term in hop.terms:
            hopping = {'relative_index': term.relative_index[0:space_size], 'from_id': term.from_id,
                       'to_id': term.to_id, 'hopping_energy': hopping_energy}
            hoppings.append(hopping)
            # if the unit cell is [0, 0]
            if np.linalg.norm(term.relative_index[0:space_size]) == 0:
                hopping = {'relative_index': term.relative_index[0:space_size], 'from_id': term.to_id,
                           'to_id': term.from_id, 'hopping_energy': np.conj(np.transpose(hopping_energy))}
                hoppings.append(hopping)
            # if the unit cell [i, j] is different than [0, 0] and also -[i, j] hoppings with opposite direction
            if np.linalg.norm(term.relative_index[0:space_size]):
                hopping = {'relative_index': -term.relative_index[0:space_size], 'from_id': term.to_id,
                           'to_id': term.from_id, 'hopping_energy': np.conj(np.transpose(hopping_energy))}
                hoppings.append(hopping)

    return {'position': position, 'hoppings': hoppings, 'num_orbitals': num_orbitals, 'space_size': space_size,
            'latt_vectors': vectors}


def define_type(precision, is_complex):
    htype = np.float32
    if is_complex == 0:
        if precision == 0:
            htype = np.float32
        elif precision == 1:
            htype = np.float64
        elif precision == 2:
            htype = np.float128
        else:
            raise SystemExit('Precision should be 0, 1 or 2')
    else:
        if precision == 0:
            htype = np.complex64
        elif precision == 1:
            htype = np.complex128
        elif precision == 2:
            htype = np.complex256
    return htype
