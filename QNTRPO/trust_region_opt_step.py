# Copyright (C) 2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import sys

import numpy as np
import torch
from torch.autograd import Variable

# sys.path.append("./Second_Order_Method")
from trust_region_opt_torch import *
from utils_trpo import *


def update_params_trust_region_step(model, get_loss, get_kl, max_kl, damping):

    """

    trust_region_step: second order trpo step
    model: torch model
    get_kl: method kl divergence
    max_kl : maximum kl_divergence
    damping: scalar parameter to make conjugate gradient method more stable -- refer to CG literature for more clarity.

    """

    loss = get_loss()
    grads = torch.autograd.grad(loss, model.parameters())
    loss_grad = torch.cat([grad.view(-1) for grad in grads]).data

    # ---- Get the current model parameters

    curr_params = get_flat_params_from(model)

    Dogleg_method_object = TR_Optimizer(
        model, get_loss, get_kl, damping, max_kl, 1e-3, 10, 1
    )  ## Create the trust region optimization object

    new_params = Dogleg_method_object.solve()

    # new_params=torch.from_numpy(new_params)
    # new_params=new_params.view(-1)
    set_flat_params_to(model, new_params)

    new_params = get_flat_params_from(model)

    print("L2 norm of change in model parameters......", np.linalg.norm(curr_params.numpy() - new_params.numpy()))

    # print ("New step with improvement in Optimization function,...", success)

    return loss
