# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.update.MuVT."""

import hoomd
import pytest
import hoomd.hpmc.pytest.conftest

# note: The parameterized tests validate parameters so we can't pass in values
# here that require preprocessing
valid_constructor_args = [
    dict(trigger=hoomd.trigger.Periodic(10),
         transfer_types=['A'],
         max_volume_rescale=0.2,
         volume_move_probability=0.5),
    dict(trigger=hoomd.trigger.After(100), transfer_types=['A', 'B']),
]

valid_attrs = [('trigger', hoomd.trigger.Periodic(10000)),
               ('trigger', hoomd.trigger.After(100)),
               ('trigger', hoomd.trigger.Before(12345)),
               ('volume_move_probability', 0.2), ('max_volume_rescale', 0.42),
               ('transfer_types', ['A']), ('transfer_types', ['B']),
               ('transfer_types', ['A', 'B'])]


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(device, constructor_args):
    """Test that MuVT can be constructed with valid arguments."""
    muvt = hoomd.hpmc.update.MuVT(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(muvt, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach(device, simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args, valid_args):
    """Test that MuVT can be attached with valid arguments."""
    integrator = valid_args[0]
    args = valid_args[1]
    n_dimensions = valid_args[2]
    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator(23456)
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"].to_base()
    mc = integrator(23456)
    mc.shape["A"] = args
    mc.shape["B"] = args

    muvt = hoomd.hpmc.update.MuVT(**constructor_args)
    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=['A', 'B'],
                                      dimensions=n_dimensions,
                                      d=2,
                                      L=50))
    sim.operations.updaters.append(muvt)
    sim.operations.integrator = mc

    sim.run(0)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        assert getattr(muvt, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr(device, attr, value):
    """Test that MuVT can get and set attributes."""
    muvt = hoomd.hpmc.update.MuVT(trigger=hoomd.trigger.Periodic(10),
                                  transfer_types=['A'])

    setattr(muvt, attr, value)
    assert getattr(muvt, attr) == value


@pytest.mark.serial
@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_attached(device, attr, value, simulation_factory,
                                two_particle_snapshot_factory, valid_args):
    """Test that MuVT can get and set attributes while attached."""
    integrator = valid_args[0]
    args = valid_args[1]
    n_dimensions = valid_args[2]
    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator(23456)
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"].to_base()
    mc = integrator(23456)
    mc.shape["A"] = args
    mc.shape["B"] = args

    muvt = hoomd.hpmc.update.MuVT(trigger=hoomd.trigger.Periodic(10),
                                  transfer_types=['A'])
    sim = simulation_factory(
        two_particle_snapshot_factory(particle_types=['A', 'B'],
                                      dimensions=n_dimensions,
                                      d=2,
                                      L=50))
    sim.operations.updaters.append(muvt)
    sim.operations.integrator = mc

    sim.run(0)

    setattr(muvt, attr, value)
    assert getattr(muvt, attr) == value


def test_insertion_removal(device, simulation_factory,
                           lattice_snapshot_factory):
    """Test that MuVT is able to insert and remove particles."""
    sim = simulation_factory(
        lattice_snapshot_factory(particle_types=['A', 'B'],
                                 dimensions=3,
                                 a=4,
                                 n=7,
                                 r=0.1))

    mc = hoomd.hpmc.integrate.Sphere(default_d=0.1, default_a=0.1)
    mc.shape['A'] = dict(diameter=1.1)
    mc.shape['B'] = dict(diameter=1.3)
    sim.operations.integrator = mc

    muvt = hoomd.hpmc.update.MuVT(trigger=hoomd.trigger.Periodic(5),
                                  transfer_types=['B'])
    sim.operations.updaters.append(muvt)

    sim.run(0)

    # we shouldn't have any particles of type B just yet
    assert (muvt.N['B'] == 0)

    # and no attempted moves
    assert sum(muvt.insert_moves) == 0
    assert sum(muvt.remove_moves) == 0

    # set a positive fugacity
    muvt.fugacity['B'] = 1

    sim.run(20)
    assert sum(muvt.insert_moves) > 0
    assert sum(muvt.remove_moves) > 0

    # make a wild guess: there be B particles
    assert (muvt.N['B'] > 0)
