# Copyright (C) 2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch


class initialize_solution_iterate(object):
    def __init__(self, x, f, g, A):

        # 		self._function_class=function_class

        self.x = x  # np.reshape(x,[len(x),1])
        self.f = f  # function_class.fun(self.x)
        self.g = g  # np.reshape(g,[len(g),1])#function_class.gradf(self.x)
        self.A = A  # function_class.trust_region_hess(self.x)

        self.error = torch.norm(self.g)
