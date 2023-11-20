# Copyright (C) 2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


"""Python script for trust region optimization using Dogleg Method"""

# from main import get_kl
#
import logging

import numpy as np
import torch

# from compute_steepest_descent_step import *
from compute_dogleg_step import *
from initialize_iterate import initialize_solution_iterate

# from lbfgs_approximation import lbfgs_approx
from quasinewton_approximation_hessian import quasinewton_approximation_torch

# from compute_trust_region_step import trpo_step
from utils_trpo import *

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


class TR_Optimizer(object):
    def __init__(self, model, f, get_kl, damping, delta0, tol, maxiter, step_type):
        """
        n_var: number of variables
        problem: class defining the problem and methods fun: for objective, gradf: for gradient of objective
        delta0: initial trust region radius
        tol: convergence tolerance
        maxiter: max number of iterations

        """

        self.model = model
        self.f = f()

        self.fval = f

        curr_params = get_flat_params_from(model)

        self._n_var = int(list(curr_params.size())[0])  # problem.get_nvar()

        self._inithessian = torch.eye(self._n_var)  # self._problem.initialize_hessian()

        ## Trust Region Hyperparamaters
        self._delta0 = delta0
        self._tol = tol
        self._maxiter = maxiter
        self._step_type = step_type

        self.damping = damping
        self.get_kl = get_kl

        ###set parameters for TR optimization

        self._parameters = {
            "tr_ratio_good": 0.75,  # if ared/pred >= ratio_good then there is possibility of TR increase
            "tr_ratio_bad": 0.1,  # if ared/pred < ratio_bad then decrease TR
            "tr_ratio_accept": 1e-4,  # if ared/pred > ratio_accept then the step is accepted
            "tr_step_factor": 0.8,  # increase TR when step is 0.8*(current TR)
            "tr_delta_small": 1e-5,  #  threshold when the TR is assumed to have become unacceptably small
            "tr_inc_delta": 2.0,  # multiplicative factor for TR when increasing
            "tr_dec_delta": 0.3,  # multiplicative factor for TR when decreasing
            "tr_maxdelta": 1 * 1e-1,  # max TR
            "tr_lbfgs": 0,  # 0: BFGS, 1:LBFGS, DONOT INDICATE 1, WE have removed lbfsgs as it wasnt included in the paper.
            "tr_lm_kmax": np.min((self._n_var, 30)),
        }

        # if 'numpy' in str(type(self._inithessian)):#len(self._inithessian)>1:
        # 	self._parameters['tr_lbfgs']=0
        # else:
        # 	self._parameters['tr_lbfgs']=1

        self._iter = 0
        self._loop = 1

        if self._parameters["tr_lbfgs"] == 1:
            self._inithessian = 1.0

        ##initialize the problem class here
        self._x0 = get_flat_params_from(model)
        self.f0 = f(True).data
        self.f0 = self.f0.item()
        self.grad0 = torch.autograd.grad(self.f, self.model.parameters(), create_graph=True, retain_graph=True)

        flat_grad = torch.cat([grad.view(-1) for grad in self.grad0]).data

        self.grad0 = flat_grad

        # print (self.grad0)
        self._trust_region_hessian0 = torch.eye(self._n_var)

        self.iterate = initialize_solution_iterate(self._x0, self.f0, self.grad0, self._trust_region_hessian0)

        ##initialize hessian approximation
        # self.hessian=quasinewton_approximation_torch(self._inithessian,self._parameters['tr_lbfgs'],self._parameters['tr_lm_kmax'])
        if self._step_type > 0:
            self.hessian = quasinewton_approximation_torch(
                self._inithessian, self._parameters["tr_lbfgs"], self._parameters["tr_lm_kmax"]
            )
        else:
            self.hessian = 0

        ##initialize trust-region step
        self.trpo_step = trpo_step(self._n_var, step_type, model, f, get_kl, damping)

        if self.iterate.error <= tol:
            self._loop = 0

        ##initialize trust-region radius
        self._delta = self._delta0

    def solve(self):

        ## compute_step
        while self._loop == 1:

            dx, dx_size, flag_step = self.trpo_step.compute_step(self.iterate, self.hessian, self._delta)  # ,Hessvec

            # x_curr=self.iterate.x
            # x_curr=x_curr.view(-1)
            # set_flat_params_to(self.model,x_curr)
            # f_curr=self.fval(True).data
            # f_curr=f_curr.numpy()
            # print('fval at current point,',f_curr)

            x_new = self.iterate.x + dx

            # xnew=torch.from_numpy(x_new)
            xnew = torch.cat([x.view(-1) for x in x_new]).data
            # xnew=xnew.view(-1)

            set_flat_params_to(self.model, xnew)
            f_new = self.fval(True).data

            print("fval at new point", f_new)
            grads = torch.autograd.grad(self.f, self.model.parameters(), create_graph=False, retain_graph=True)
            g_new = torch.cat([grad.view(-1) for grad in grads]).data

            act_dec = self.iterate.f - f_new

            if self._step_type == 0:
                a = torch.cat([grad.view(-1) for grad in self.iterate.g]).data
                pre_dec = -torch.dot(a, dx)

                # pre_dec=pre_dec.numpy()
                print("Predicted decrease and actual decrease difference", pre_dec - act_dec)
            else:
                Hdx = self.hessian.hessvec(dx)
                pre_dec = -np.vdot(self.iterate.g, dx) - 0.5 * np.vdot(dx, Hdx)
                if pre_dec <= 0:
                    logging.debug(
                        "flag = %c gdx = %e quad = %e", flag_step, np.vdot(self.iterate.g, dx), 0.5 * np.vdot(dx, Hdx)
                    )

            # ratio=act_dec/(1e-16+pre_dec)
            ratio = act_dec / (pre_dec + 1e-16)

            ## Check progress of solution
            accept_flag = 1
            if act_dec <= 0:
                print(self._iter, " act_dec=", act_dec, " pre_dec=", pre_dec)
            # 			print (act_)
            if act_dec <= 0 or ratio <= self._parameters["tr_ratio_accept"]:
                accept_flag = 0

            delta_old = self._delta
            delta_change = 0

            if act_dec >= 0 and ratio >= self._parameters["tr_ratio_good"]:

                if dx_size >= self._parameters["tr_step_factor"] * self._delta:

                    self._delta = min(self._parameters["tr_maxdelta"], self._delta * self._parameters["tr_inc_delta"])
                    delta_change = 1

            elif ratio >= self._parameters["tr_ratio_bad"] and ratio <= self._parameters["tr_ratio_good"]:
                pass  ## do nothing

            else:
                self._delta = self._delta * self._parameters["tr_dec_delta"]
                delta_change = -1

            s = dx

            # print('size of s-----',s.shape,self.iterate.g.shape)

            g_earlier = g_new

            y = g_new - self.iterate.g

            # print('size of y---------',y.shape)

            if accept_flag == 1:

                self.iterate.x = x_new
                self.iterate.f = f_new
                self.iterate.g = g_new
                # self.iterate.A=self._trust_region_hessian0#self._problem.trust_region_hess(x_new) #(self.iterate.x)
                self.iterate.error = self.compute_error(g_new)

            ### update hessian
            if self._step_type > 0:
                self.hessian.update(s, y)

            if (
                self._parameters["tr_lbfgs"] == 1
                and act_dec <= 0
                and (delta_change == 0 or self._delta <= 10 * self._parameters["tr_delta_small"])
            ):

                ## indication that there is no progress. Get rid of the vectors and start over
                self.hessian.reset()
                # resetting the trust region size if this becomes too small
                if self._delta <= 10 * self._parameters["tr_delta_small"]:
                    self._delta = self._delta0

            self._iter += 1

            ## print statistics
            nrmHess = 0.0
            if self._step_type > 0 and self._parameters["tr_lbfgs"] == 0:
                nrmHess = torch.norm(self.hessian.hessian)
            if np.mod(self._iter, 10) == 1:

                logging.debug("Iteration Objective  ||g|| Ratio ||dx|| Accept delta Change  nrm(Hess)")

            logging.debug(
                "%d %e %e %e %e%c %d %e %d %e",
                self._iter,
                self.iterate.f,
                self.iterate.error,
                ratio,
                dx_size,
                flag_step,
                accept_flag,
                delta_old,
                delta_change,
                nrmHess,
            )
            # 			logging.debug (self._iter, self.iterate.f, self.iterate.error, ratio, dx_size, accept_flag, delta_old, delta_change, nrmHess)

            if self.iterate.error <= self._tol:
                self._loop = 0

                break
            elif self._iter >= self._maxiter:
                self._loop = 1
                break
            elif self._delta <= self._parameters["tr_delta_small"] and accept_flag == 0:
                self._loop = 2
                break

        return xnew

    def compute_error(self, g):

        a = torch.norm(g)

        return a
