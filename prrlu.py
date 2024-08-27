"""
Partial rank-revealing LU decomposition. 
https://arxiv.org/abs/2407.02454, section 3.3
"""

import numpy as np
from numpy.linalg import inv
from scipy.linalg import lu
from typing import Optional


class PrrLU:
    """
    Full search prrLU algorithm.
    """

    result_L = None
    result_D = None
    result_U = None
    result_rank = None

    def __init__(self, debug: bool = False):
        self.debug = debug

    def _permut(self, old_A, top_left_ind):
        A = old_A[top_left_ind:, top_left_ind:]
        abs_A = np.abs(A)
        idx_r, idx_c = np.asarray(abs_A == np.max(abs_A)).nonzero()
        if idx_r[0] != 0:
            A[[idx_r[0], 0], :] = A[[0, idx_r[0]], :]
        if idx_c[0] != 0:
            A[:, [idx_c[0], 0]] = A[:, [0, idx_c[0]]]
        old_A[top_left_ind:, top_left_ind:] = A
        if self.debug:
            print(f"A after permut is {old_A}")
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
        shurcomp = np.where(shurcomp < 1e-15, 0, shurcomp)
        if self.debug:
            print(f"second is {second} and compliment is {shurcomp}")
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

        return L, D, U

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
        if step_num < 1:
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
                    f"Matrix L is {L_new}\n Matrix D is {D_new}\n Matrix U is {U_new}",
                )

            L, U = L @ L_new, U @ U_new
            A = D_new

            if print_steps:
                print(
                    f"LDU decompositions on step {step}:\n",
                    f"Matrix L is {L}\n Matrix D is {A}\n Matrix U is {U}",
                )

            if (shur_compliment == 0).all():
                self.result_rank = step
                self.result_D = A
                self.result_L = L
                self.result_U = U
                print(
                    f"Decomposition with max rank was found. Matrix rank is {self.result_rank}.\n",
                    f"Matrix L is {L}\n Matrix D is {A}\n Matrix U is {U}",
                )
                return L, A, U, step

        print(
            f"Decomposition with rank {step_num} was found.\n",
            f"Matrix L is {L}\n Matrix D is {A}\n Matrix U is {U}",
        )
        return L, A, U, step_num
