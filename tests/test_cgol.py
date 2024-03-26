"""Testing suite for cgol.py"""

import os
import hypothesis
import pytest

import numpy as np

from hypothesis import strategies as st
from hypothesis.extra import numpy as henp

from cgol import cgol

@hypothesis.given(shape=henp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=2))
def test_world_is_ndarray(shape):
    """assert world is a numpy array"""
    world = cgol.init_world(shape)
    assert type(world) == np.ndarray


@hypothesis.given(shape=henp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=2))
def test_world_has_correct_shape(shape):
    """assert world initialises with correct shape"""
    world = cgol.init_world(shape)
    assert world.shape == shape


@hypothesis.given(shape=henp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=2))
def test_world_dytype_is_bool(shape):
    """assert world cells are either alive (True) or dead (False)"""
    world = cgol.init_world(shape)
    assert world.dtype == bool

@pytest.fixture()
def glider_world():
    world = cgol.init_world([9, 9])
    world.fill(False)
    glider = glider = np.array([[0,1,0],[0,0,1],[1,1,1]])
    world[3:6,3:6] = glider
    return world


@pytest.fixture()
def glider_neighbours():
    array = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 1., 1., 1., 0., 0., 0.],
                      [0., 0., 0., 1., 1., 2., 1., 0., 0.],
                      [0., 0., 1., 3., 5., 3., 2., 0., 0.],
                      [0., 0., 1., 1., 3., 2., 2., 0., 0.],
                      [0., 0., 1., 2., 3., 2., 1., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    return array


def test_neighbours(glider_world, glider_neighbours):
    """check that all neighbours are found"""
    assert np.equal(cgol.neighbours(glider_world), glider_neighbours).all()

@pytest.fixture()
def blinker():
    "oscillator with periode 2"
    blinker = np.zeros([3,3])
    blinker[1] = 1
    return blinker


def test_update_blinker(blinker):
    clone = blinker.copy()
    ccc = cgol.update(blinker)
    assert np.equal(ccc, clone.T).all()


def test_update_blinker_twice(blinker):
    """
    blinker has period 2, so updating a blinker twice 
    reverts it back to it's original state
    """
    clone = blinker.copy()
    assert np.equal(cgol.update(cgol.update(blinker)), clone).all()

DATA = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../../data"
)
PATTERNS = pytest.mark.datafiles(
    os.path.join(DATA, "blinker.csv"),
    os.path.join(DATA, "pulsar.csv"),
)

@PATTERNS
def test_load(blinker, datafiles):
    loaded_blinker = cgol.load(datafiles / "blinker.csv")
    assert np.equal(blinker, loaded_blinker).all()


@PATTERNS
def test_run_blinker_period_1_fails(datafiles):
    """blinkers have period 2, hence a cycle of 1 should fail"""
    blinker = np.loadtxt(datafiles / "blinker.csv", delimiter=",").astype(bool)
    clone = blinker.copy()
    assert  not np.equal(cgol._run(clone, cycles=1), blinker).all()


@PATTERNS
def test_run_blinker_period_2_succeeds(datafiles):
    """blinkers have period 2, hence a cycle of 2 should succeed"""
    blinker = np.loadtxt(datafiles / "blinker.csv", delimiter=",").astype(bool)
    clone = blinker.copy()
    assert np.equal(cgol._run(clone, cycles=2), blinker).all()


@PATTERNS
def test_run_blinker_period_3_fails(datafiles):
    """pulsars have period 3, hence a cycle of 1 should fail"""
    pulsar = np.loadtxt(datafiles / "pulsar.csv", delimiter=",").astype(bool)
    clone = pulsar.copy()
    assert not np.equal(cgol._run(clone, cycles=1), pulsar).all()


@PATTERNS
def test_run_blinker_period_3_succeeds(datafiles):
    """pulsars have period 3, hence a cycle of 3 should succeed"""
    pulsar = np.loadtxt(datafiles / "pulsar.csv", delimiter=",").astype(bool)
    clone = pulsar.copy()
    assert np.equal(cgol._run(clone, cycles=3), pulsar).all()
