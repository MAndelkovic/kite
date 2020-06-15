import pybinding as pb


def haldane():
    """Return the lattice specification for Haldane model"""
    from math import sqrt
    a = 0.24595  # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance

    t = -1
    t2 = t / 10
    m = 0
    # create a lattice with 2 primitive vectors
    lat = pb.Lattice(
        a1=[a, 0],
        a2=[a / 2, a / 2 * sqrt(3)]
    )

    lat.add_sublattices(
        # name and position
        ('A', [0, -a_cc / 2], -m),
        ('B', [0, a_cc / 2], m)
    )

    lat.add_hoppings(
        # inside the main cell
        ([0, 0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t),
        ([1, 0], 'A', 'A', t2 * 1j),
        ([0, -1], 'A', 'A', t2 * 1j),
        ([-1, 1], 'A', 'A', t2 * 1j),
        ([1, 0], 'B', 'B', t2 * -1j),
        ([0, -1], 'B', 'B', t2 * -1j),
        ([-1, 1], 'B', 'B', t2 * -1j)
    )

    return lat
