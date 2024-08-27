import pytest
import numpy as np
from methods.prrlu import PrrLU


@pytest.mark.matrix
def test_A():
    A = np.array([[2, 4], [3, 5]])
    c = PrrLU()
    L, D, U, chi = c.find_decomposition(A)
    print(D)
    L_check = np.array([[1, 0], [4 / 5, 1]])
    D_check = np.array([[5, 0], [0, -2 / 5]])
    U_check = np.array([[1, 3 / 5], [0, 1]])
    chi_check = 2
    assert np.all(L == L_check), f"L doesn't match reference matrix: {L} != {L_check}"
    assert np.all(D == D_check), f"D doesn't match reference matrix: {D} != {D_check}"
    assert np.all(U == U_check), f"U doesn't match reference matrix: {U} != {U_check}"
    assert (
        chi == chi_check
    ), f"chi doesn't match reference chi check {chi} != {chi_check}"


@pytest.mark.matrix
def test_B():
    B = np.array([[1, 2, 3], [2, 4, 6], [4, 8, 12]])
    c = PrrLU()
    L, D, U, chi = c.find_decomposition(B)


@pytest.mark.matrix
def test_C():
    C = np.array([[2, 3, 4], [3, 4, 5], [1, 5, 9]])
    c = PrrLU()
    L, D, U, chi = c.find_decomposition(C)    
