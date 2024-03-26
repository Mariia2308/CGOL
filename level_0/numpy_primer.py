"""A primer on NumPy"""

import numpy as np

def is_odd(number):
    """Returns True if the number is odd, False if the number is even.

    Args:
        number (int): The number to be tested for oddity.

    Returns:
        bool: True if the number is odd, False if the number is even.

    Example:
        >>> is_odd(42)
        False
        >>> is_odd(23)
        True
    """
    return number % 2 != 0

def square_odd_square_matrix(n):
    """Generates an nxn matrix with ascending numbers from 0 to (n**2 - 1) where the odd numbers are squared
    and the even numbers are zero.

    Args:
        n (int): the number of rows and columns of the matrix

    Returns:
        (np.array): an n x n matrix with the odd numbers squared and even numbers as 0.

    Example:
        >>> square_odd_square_matrix(3)
        array([[0, 1, 0],
               [9, 0, 25],
               [0, 49, 0]])
        >>> square_odd_square_matrix(4).shape
        (4, 4)
    """
    matrix = np.arange(n**2).reshape(n, n)
    matrix[matrix % 2 != 0] **= 2  
    matrix[matrix % 2 == 0] = 0    
    return matrix


def east(matrix):
    """shift the columns of the matrix one to the right

    Args:
        matrix (np.array):  a 2-d matrix

    Returns:
        (np.array):  a 2-d matrix with each column shifted to the right,
                     dropping the last column, padding the first with zeros.
    Example:
        >>> a = np.array([[1, 2, 3],
                          [4, 5, 6]])
        >>> east(a)
        <<< array([[0, 1, 2],
                   [0, 4, 5]])
    """
    rows, cols = matrix.shape
    result = np.zeros_like(matrix) 
    result[:, 1:] = matrix[:, :-1]  
    return result


def west(matrix):
    """shift the columns of the matrix one to the left

    Args:
        matrix (np.array):  a 2-d matrix

    Returns:
        (np.array):  a 2-d matrix with each column shifted to the left,
                     dropping the first column, padding the last with zeros.
    Example:
        >>> a = np.array([[1, 2, 3],
                          [4, 5, 6]])
        >>> west(a)
        <<< array([[2, 3, 0],
                   [5, 6, 0]])
    """
    rows, cols = matrix.shape
    result = np.zeros_like(matrix) 
    result[:, :-1] = matrix[:, 1:]  
    return result


def count_occurrences_of_elements_divisible_by_5(matrix):
    """counts the number of elements in matrix that are divisible by 5

    Args:
        matrix (np.array): the matrix whose elements we want to analyse

    Returns:
        (int): the number of elements that are divisible by 5 (no rest)
    Example:
        >>> count_occurrences_of_elements_divisible_by_5(np.array(5, 3, 5))
        <<< 2
    """
    divisible_by_5 = matrix % 5 == 0
    count = np.sum(divisible_by_5)
    return count
