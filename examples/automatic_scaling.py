""" Automatic scaling and pybinding model

    Lattice : Monolayer graphene;
    Disorder : Disorder class Deterministic and Uniform at different sublattices,
               StructuralDisorder class vacancy and bond disorder;
    Configuration : size of the system 256x256, without domain decomposition (nx=ny=1), periodic boundary conditions,
                    double precision, manual scaling;
    Calculation : dos;
    Modification : magnetic field is off;

    Note : automatic scaling is not supported when bond disorder is present!
"""

import kite
import matplotlib.pyplot as plt
import numpy as np
import pybinding as pb

from pybinding.repository import graphene

# load a monolayer graphene lattice
lattice = graphene.monolayer()
# add Disorder
disorder = kite.Disorder(lattice)
disorder.add_disorder('B', 'Deterministic', -0.3)
disorder.add_disorder('A', 'Uniform', +0.5, 0.5)
# add vacancy StructuralDisorder
disorder_struc = kite.StructuralDisorder(lattice, concentration=0.05)
disorder_struc.add_vacancy('A')
# number of decomposition parts in each direction of matrix. This divides the lattice into various sections,
# each of which is calculated in parallel
nx = ny = 2
# number of unit cells in each direction.
lx = ly = 256
# make config object which caries info about
# - the number of decomposition parts [nx, ny],
# - lengths of structure [lx, ly]
# - boundary conditions, setting True as periodic boundary conditions, and False elsewise,
# - info if the exported hopping and onsite data should be complex,
# - info of the precision of the exported hopping and onsite data, 0 - float, 1 - double, and 2 - long double.
# - scaling, if None it's automatic, if present select spectrum_bound=[e_min, e_max]
configuration = kite.Configuration(divisions=[nx, ny], length=[lx, ly], boundaries=[True, True], is_complex=False,
                                   precision=1)

# manual scaling
# configuration = kite.Configuration(divisions=[nx, ny], length=[lx, ly], boundaries=[True, True],
#                                    is_complex=False, precision=1, spectrum_range=[-10, 10])

# require the calculation of DOS
calculation = kite.Calculation(configuration)
calculation.dos(num_points=10000, num_moments=1024, num_random=1, num_disorder=1)
# configure the *.h5 file
kite.config_system(lattice, configuration, calculation, filename='auto_scaling.h5',
                   disorder=disorder, disorder_structural=disorder_struc)

# make a rough estimate of KITE model using Pybinding,
# (more disorder realisations you average, more similar it will be)
# calculate DOS to confirm
model = kite.make_pybinding_model(lattice, disorder=disorder, disorder_structural=disorder_struc)
kpm = pb.kpm(model)
dos = kpm.calc_dos(energy=np.linspace(-10, 10, 10000), broadening=1e-2, num_random=1)
dos.plot()
plt.show()