"""Testing suite of numpy_primer"""

import os

import hypothesis
import pytest

import numpy as np

from hypothesis import strategies as st
from hypothesis.extra import numpy as henp

from level_0 import numpy_primer as primer


@hypothesis.given(number=st.integers().filter(lambda x: x % 2))
def test_is_odd_is_true(number):
    """assert that odd numbers are marked as True"""
    assert primer.is_odd(number)


@hypothesis.given(number=st.integers().filter(lambda x: not x % 2))
def test_is_odd_is_false(number):
    """assert that even numbers are marked as False"""
    assert not primer.is_odd(number)

@hypothesis.given(number=st.integers(min_value=2, max_value=100))
def test_square_odd_square_matrix_shape(number):
    """assert a square matrix of dimension number x number"""
    assert primer.square_odd_square_matrix(number).shape == (number, number)

@hypothesis.given(number=st.integers(min_value=2, max_value=100))
def test_square_odd_square_matrix_elements_are_odd_or_0(number):
    """assert each element in the matrix is odd or 0

    Note: this test recycles the primer.is_odd function to broadcast over a test array. You should not do this in general as it couples tests together.
    """
    array = primer.square_odd_square_matrix(number)
    non_zero_elements = array[array != 0]
    assert all(non_zero_elements % 2 != 0)


@hypothesis.given(number=st.integers(min_value=2, max_value=100))
def test_square_odd_square_matrix_elements_are_integers(number):
    """assert that the elements of the matrix are all integers"""
    assert primer.square_odd_square_matrix(number).dtype == int


@pytest.fixture
def sosm():
    """a 10 x 10 square odd square matrix"""
    array = np.array([[ 0,  1,  0,  3,  0,  5,  0,  7,  0,  9],
                      [ 0, 11,  0, 13,  0, 15,  0, 17,  0, 19],
                      [ 0, 21,  0, 23,  0, 25,  0, 27,  0, 29],
                      [ 0, 31,  0, 33,  0, 35,  0, 37,  0, 39],
                      [ 0, 41,  0, 43,  0, 45,  0, 47,  0, 49],
                      [ 0, 51,  0, 53,  0, 55,  0, 57,  0, 59],
                      [ 0, 61,  0, 63,  0, 65,  0, 67,  0, 69],
                      [ 0, 71,  0, 73,  0, 75,  0, 77,  0, 79],
                      [ 0, 81,  0, 83,  0, 85,  0, 87,  0, 89],
                      [ 0, 91,  0, 93,  0, 95,  0, 97,  0, 99]])
    return array

@pytest.fixture
def sosm_east():
    """a 10 x 10 square odd square matrix"""
    array = np.array([[ 0,  0,  1,  0,  3,  0,  5,  0,  7,  0],
                      [ 0,  0, 11,  0, 13,  0, 15,  0, 17,  0],
                      [ 0,  0, 21,  0, 23,  0, 25,  0, 27,  0],
                      [ 0,  0, 31,  0, 33,  0, 35,  0, 37,  0],
                      [ 0,  0, 41,  0, 43,  0, 45,  0, 47,  0],
                      [ 0,  0, 51,  0, 53,  0, 55,  0, 57,  0],
                      [ 0,  0, 61,  0, 63,  0, 65,  0, 67,  0],
                      [ 0,  0, 71,  0, 73,  0, 75,  0, 77,  0],
                      [ 0,  0, 81,  0, 83,  0, 85,  0, 87,  0],
                      [ 0,  0, 91,  0, 93,  0, 95,  0, 97,  0]])
    return array


@pytest.fixture
def sosm_west():
    """a 10 x 10 square odd square matrix"""
    array = np.array([[ 1,  0,  3,  0,  5,  0,  7,  0,  9,  0],
                      [11,  0, 13,  0, 15,  0, 17,  0, 19,  0],
                      [21,  0, 23,  0, 25,  0, 27,  0, 29,  0],
                      [31,  0, 33,  0, 35,  0, 37,  0, 39,  0],
                      [41,  0, 43,  0, 45,  0, 47,  0, 49,  0],
                      [51,  0, 53,  0, 55,  0, 57,  0, 59,  0],
                      [61,  0, 63,  0, 65,  0, 67,  0, 69,  0],
                      [71,  0, 73,  0, 75,  0, 77,  0, 79,  0],
                      [81,  0, 83,  0, 85,  0, 87,  0, 89,  0],
                      [91,  0, 93,  0, 95,  0, 97,  0, 99,  0]])
    return array


@hypothesis.given(
    matrix=henp.arrays(
        dtype=int,
        shape=henp.array_shapes(min_dims=2, max_dims=2, min_side=2)
    )
)
def test_east_pads_first_col_with_zeros(matrix):
    """assures east results in a first column of zeros"""
    assert not primer.east(matrix)[:,0].any()


@hypothesis.given(
    matrix=henp.arrays(
        dtype=int,
        shape=henp.array_shapes(min_dims=2, max_dims=2, min_side=2)
    )
)
def test_west_pads_last_with_zeros(matrix):
    """assures west results in a last column of zeros"""
    assert not primer.west(matrix)[:,-1].any()


def test_east_shifts_to_the_right(sosm, sosm_east):
    assert np.equal(primer.east(sosm), sosm_east).all()


def test_west_shifts_to_the_left(sosm, sosm_west):
    assert np.equal(primer.west(sosm), sosm_west).all()


def test_count_occurrences_of_elements_divisible_by_5(sosm, sosm_west, sosm_east):
    assert primer.count_occurrences_of_elements_divisible_by_5(sosm) == 60
    assert primer.count_occurrences_of_elements_divisible_by_5(sosm_east) == 70
    assert primer.count_occurrences_of_elements_divisible_by_5(sosm_west) == 60


@pytest.fixture()
def glider():
    array = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [1, 1, 1]])
    return array

DATA = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../../data"
)
PATTERNS = pytest.mark.datafiles(
    os.path.join(DATA, "glider.csv"),
)

@PATTERNS
def test_glider_transpose(glider, datafiles):
    array = np.loadtxt(datafiles / "glider.csv", delimiter=",")
    assert np.equal(array, glider).all()

