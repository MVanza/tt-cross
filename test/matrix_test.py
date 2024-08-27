import numpy as np
from methods.prrlu import PrrLU


def test_A():
    A = np.array([[2, 4], [3, 5]])
    c = PrrLU()
    L, D, U, chi = c.find_decomposition(A)
    L_check = np.array([[1, 0], [4 / 5, 1]])
    D_check = np.array([[5, 0], [0, -2 / 5]])
    U_check = np.array([[1, 3 / 5], [0, 1]])
    chi_check = 2
    assert L == L_check, f"L doesn't match reference matrix: {L} != {L_check}"
    assert D == D_check, f"D doesn't match reference matrix: {D} != {D_check}"
    assert U == U_check, f"U doesn't match reference matrix: {U} != {U_check}"
    assert (
        chi == chi_check
    ), f"chi doesn't match reference chi check {chi} != {chi_check}"


def test_B():
    B = np.array([[1, 2, 3], [2, 4, 6], [4, 8, 12]])


def test_C():
    C = np.array([[2, 3, 4], [3, 4, 5], [1, 5, 9]])
