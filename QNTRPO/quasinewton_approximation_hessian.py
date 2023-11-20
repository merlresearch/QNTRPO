# Copyright (C) 2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch


class quasinewton_approximation_torch(object):
    def __init__(self, inithessian, approx_type, memory=30):

        self.inithessian = inithessian  # initial hessian matrix for BFGS, scalar multiplying identity for L-BFGS
        self.type = approx_type  # 0: BFGS, 1: L-BFGS
        self.memory = memory  # indicates number of vectors that are stored for L-BFGS

        if self.type == 0:  # BFGS
            self.hessian = inithessian
        if self.type == 1:  # L-BFGS
            self.delta = inithessian
            self.m = 0  # number of vectors currently in the limited memory approximation
            self.S = 0  # this is the collection of s vectors
            self.Y = 0  # this is the collection of y vectors
            self.Minv = 0  # this is the matrix in the middle of the limited memory approximation
            self.L = 0  # this is the lower triangular matrix
            self.d = 0  # this is the diagonal of the SE submatrix in M
            self.STS = 0  # this is the diagonal of the SE submatrix in M

    def update(self, s, y):

        ys = torch.dot(y, s)
        if ys <= 1e-3:
            return
        rho = 1.0 / torch.dot(y, s)

        print("rho", rho)

        if self.type == 0:  # BFGS
            Hess_s = torch.matmul(self.hessian, s)
            sT_Hess_s = torch.dot(s, Hess_s)
            self.hessian = self.hessian + rho * torch.ger(y, y) - 1 / (sT_Hess_s) * torch.ger(Hess_s, Hess_s)

    def hessvec(self, x):

        if self.type == 0:  # BFGS
            Hess_x = torch.matmul(self.hessian, x)
            return Hess_x

        if self.type == 1:  # L-BFGS
            if self.m == 0:
                Hess_x = self.delta * x
                print("Hess_x", Hess_x.size())
                return Hess_x

            # TODO: pythonify this
            x1 = torch.cat(
                (self.delta * torch.matmul(torch.t(self.S), x), torch.matmul(torch.t(self.Y), x)), 0
            )  # np.vstack((self.delta*np.matmul(self.S.T,x),np.matmul(self.Y.T,x)))
            x2 = torch.matmul(self.Minv, x1)
            x3 = self.delta * torch.matmul(self.S, x2[0 : self.m]) + torch.matmul(self.Y, x2[self.m :])
            Hess_x = self.delta * x - x3

            print("Hess_x", Hess_x.size())
            return Hess_x

    def reset(self):

        if self.type == 1:  # L-BFGS
            self.delta = self.delta
            self.m = 0  # number of vectors currently in the limited memory approximation
            self.S = 0  # this is the collection of s vectors
            self.Y = 0  # this is the collection of y vectors
            self.Minv = 0  # this is the matrix in the middle of the limited memory approximation
            self.L = 0  # this is the matrix in the middle of the limited memory approximation
            self.d = 0  # this is the diagonal of the SE submatrix in M
            self.STS = 0  # this is the diagonal of the SE submatrix in M
