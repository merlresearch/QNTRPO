# Copyright (C) 2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Python script to compute the trust region step """

import logging

import numpy as np
import torch
from torch.autograd import Variable

logging.basicConfig(level=logging.DEBUG)

from conjugate_grad_solution import conjugate_gradient

logger = logging.getLogger(__name__)


class trpo_step(object):
    def __init__(self, n_var, type_option, model, f, get_kl, damping):

        self._n_var = n_var
        self.type = type_option
        self.model = model
        self.f = f
        self.get_kl = get_kl
        self.damping = damping

    # 		self.conjugate_gradient_solver=conjugate_gradients

    def compute_step(self, iterate, hessian, delta):

        if self.type == 0:

            ## Compute scaled steepest descent #### DO CG
            # [Ainvg,flag_Ag] = self.conjugate_gradient(iterate.A,iterate.g,1e-6,2*self._n_var,1)
            a = torch.cat([grad.view(-1) for grad in iterate.g]).data

            Ainvg, flag_Ag = conjugate_gradient(self.Fvp, -a, 100, 1e-6)  # 2*self._n_var
            # Ainvg=Ainvg.numpy()
            # flag_Ag=flag_Ag.numpy()

            print("flag_Ag", flag_Ag)

            if flag_Ag == 0:
                print("In the scaling loop")

                vecpdt = torch.dot(Ainvg, -a)

                scaledx = delta / torch.sqrt(vecpdt)

                print("Scale dx", scaledx)
                dx = scaledx * Ainvg  # scaledx*Ainvg.numpy()
            else:
                ##### Do FVP
                Fvp = iterate.A
                b = torch.from_numpy(iterate.g)
                b = b.view(-1)
                Ag = Fvp(b)
                Ag = Ag.numpy()
                # Ag = np.matmul(iterate.A,iterate.g)
                scaledx = delta / np.sqrt(np.vdot(iterate.g, Ag))
                dx = -scaledx * iterate.g

            dx_size = delta
            flag = "S"

            return dx, dx_size, flag

    def Fvp(self, v):

        """
        function to compute the fisher vector product , i.e., Ainv*vec
        where Ainv is the inverse of the FIM

        """
        model = self.model
        get_kl = self.get_kl
        damping = self.damping

        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()

        grads = torch.autograd.grad(kl_v, model.parameters())

        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping
