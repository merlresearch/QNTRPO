# Copyright (C) 2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch


def conjugate_gradient_fullmat(Avp, b, nsteps, residual_tol=1e-6):
    """
    Returns F^(-1)b where F is the Hessian of the KL divergence
    """
    p = b.clone().data
    r = b.clone().data
    x = np.zeros_like(b.data.cpu().numpy())

    x = torch.from_numpy(x)

    flag_cg = 1
    rdotr = torch.dot(r, r)  # r.double().dot(r.double())
    for _ in range(nsteps):
        z = Avp.hessvec(p)  # self.hessian_vector_product(Variable(p)).squeeze(0)
        v = rdotr / torch.dot(p, z)  # p.double().dot(z.double())
        x += v * p  # (p.cpu().numpy())
        r -= v * z
        newrdotr = torch.dot(r, r)  # r.double().dot(r.double())
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            flag_cg = 0
            break
    return x, flag_cg
