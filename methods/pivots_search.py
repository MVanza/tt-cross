"""
Partial rank-revealing LU decomposition. 
https://arxiv.org/abs/2407.02454, section 3.3
"""

import numpy as np
from numpy.linalg import inv, LinAlgError
from scipy.linalg import lu
from typing import Optional
from random import seed, randint

DECS = 14


def trunc(values):
    return np.trunc(values * 10**DECS) / (10**DECS)


class FullSearch:
    """
    Full search prrLU algorithm.
    """

    result_L = None
    result_D = None
    result_U = None
    result_rank = None

    pivot_set = {}

    def __init__(self, inverse_need: bool = False, debug: bool = False):
        self.debug = debug
        self.inverse_need = inverse_need
        self.r_permut_list = []
        self.c_permut_list = []

    def _permut(self, old_A, top_left_ind):
        if not self.r_permut_list:
            self.r_permut_list = list(range(old_A.shape[0]))
        if not self.c_permut_list:
            self.c_permut_list = list(range(old_A.shape[1]))
        A = old_A[top_left_ind:, top_left_ind:]
        idx_r, idx_c = np.asarray(np.abs(A) == np.max(np.abs(A))).nonzero()
        if idx_r[0] != 0:
            A[[idx_r[0], 0], :] = A[[0, idx_r[0]], :]
            (
                self.r_permut_list[idx_r[0] + top_left_ind],
                self.r_permut_list[0 + top_left_ind],
            ) = (
                self.r_permut_list[0 + top_left_ind],
                self.r_permut_list[idx_r[0] + top_left_ind],
            )
        if idx_c[0] != 0:
            A[:, [idx_c[0], 0]] = A[:, [0, idx_c[0]]]
            (
                self.c_permut_list[idx_c[0] + top_left_ind],
                self.c_permut_list[0 + top_left_ind],
            ) = (
                self.c_permut_list[0 + top_left_ind],
                self.c_permut_list[idx_c[0] + top_left_ind],
            )
        old_A[top_left_ind:, top_left_ind:] = A

        if self.debug:
            print(f"A after permut is\n{old_A}")
            print(
                f"Permut list for row is {self.r_permut_list} and for column {self.c_permut_list}"
            )
        return old_A

    def _get_shur_comp(self, A11, A12, A21, A22):
        if self.debug:
            print(f"blocks is A11 = {A11}, A12 = {A12}, A21 = {A21} and A22 = {A22}")
        if A11.shape == (1, 1):
            second = np.outer((A21 * 1 / A11[0]), A12)
        else:
            if A12.ndim == 0:
                second = A21 * inv(A11) * A12
            else:
                second = A21 @ inv(A11) @ A12
        shurcomp = A22 - second
        shurcomp = np.where(np.abs(shurcomp) < 1e-10, 0, shurcomp)
        if self.debug:
            print(f"second is \n{second} and compliment is \n{shurcomp}")
        return shurcomp

    def _ldu_decompose(self, A11, A12, A21, A22, shur_compliment):
        if A11.shape != (1, 1):
            inv_A11 = inv(A11)
            L = np.block(
                [
                    [np.eye(N=A11.shape[0], M=A11.shape[1]), np.zeros(A12.shape)],
                    [A21 @ inv_A11, np.eye(N=A22.shape[0], M=A22.shape[1])],
                ]
            )
            D = np.block(
                [[A11, np.zeros(A12.shape)], [np.zeros(A21.shape), shur_compliment]]
            )
            U = np.block(
                [
                    [np.eye(N=A11.shape[0], M=A11.shape[1]), inv_A11 @ A12],
                    [np.zeros(A21.shape), np.eye(N=A22.shape[0], M=A22.shape[1])],
                ]
            )
        else:
            inv_A11 = 1 / A11[0]
            L = np.block(
                [
                    [1, np.zeros(A12.shape)],
                    [A21 * inv_A11, np.eye(N=A22.shape[0], M=A22.shape[1])],
                ]
            )
            D = np.block(
                [[A11, np.zeros(A12.shape)], [np.zeros(A21.shape), shur_compliment]]
            )
            U = np.block(
                [
                    [1, inv_A11 * A12],
                    [np.zeros(A21.shape), np.eye(N=A22.shape[0], M=A22.shape[1])],
                ]
            )
        L = trunc(L)
        D = trunc(D)
        U = trunc(U)
        return L, D, U

    def get_pivots(self):
        return self.pivot_set

    def find_decomposition(
        self, matrix, step_num: Optional[int] = None, print_steps=False
    ):
        A = matrix.copy()
        if A.ndim < 2:
            raise ValueError("Scalar and vectors are not allowed")
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix should be square")
        if step_num is None:
            step_num = A.shape[0]
        if step_num <= 1:
            raise ValueError("Incorrect chi parameter")
        L = np.eye(A.shape[0])
        U = np.eye(A.shape[0])
        for step in range(1, step_num):

            A = self._permut(A, step - 1)

            # get blocks
            A11 = A[:step, :step]
            A22 = A[step:, step:]
            A12 = A[:step, step:]
            A21 = A[step:, :step]

            shur_compliment = self._get_shur_comp(A11, A12, A21, A22)

            L_new, D_new, U_new = self._ldu_decompose(
                A11, A12, A21, A22, shur_compliment
            )

            if self.debug:
                print(
                    f"Matrices after ldu decompositions on step {step}:\n",
                    f"Matrix L is \n{L_new}\n Matrix D is \n{D_new}\n Matrix U is \n{U_new}",
                )

            L = L @ L_new
            U = U_new @ U
            A = D_new

            if print_steps:
                print(
                    f"LDU decompositions on step {step}:\n",
                    f"Matrix L is\n{L}\n Matrix D is\n{A}\n Matrix U is\n{U}",
                )

            if (shur_compliment == 0).all():
                self.result_rank = step
                self.result_D = A
                self.result_L = L
                self.result_U = U
                print(
                    f"Decomposition with max rank was found. Matrix rank is {self.result_rank}.\n",
                    f"Matrix L is\n{L}\n Matrix D is\n{A}\n Matrix U is\n{U}",
                )
                self.pivot_set = {
                    "I": self.r_permut_list[:step],
                    "J": self.c_permut_list[:step],
                }
                if self.debug:
                    print(
                        f"row pivots is {self.pivot_set['I']} and column pivots is {self.pivot_set['J']}"
                    )
                return L, A, U, step

        print(
            f"Decomposition with rank {step_num} was found.\n",
            f"Matrix L is\n{L}\n Matrix D is\n{A}\n Matrix U is\n{U}",
        )
        self.pivot_set = {
            "I": self.r_permut_list[: step_num - 1],
            "J": self.c_permut_list[: step_num - 1],
        }
        if self.debug:
            print(
                f"row pivots is {self.pivot_set['I']} and column pivots is {self.pivot_set['J']}"
            )
        return L, A, U, step_num
