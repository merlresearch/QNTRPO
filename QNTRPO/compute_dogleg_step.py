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
from conjugate_grad_solution_fullmat import conjugate_gradient_fullmat

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

        if self.type == 1:
            ## Implement Dogleg method

            """

            get the Newton direction, check to see if it is within the trust-region
            original QP: min g^T d + 0.5*d^T*H*d s.t. d^T A d <= delta^2
            transformed QP: min ghat^T dhat + 0.5*dhat^T*Hhat*dhat s.t. dhat^T dhat <= delta^2
            where A = L*L^T, dhat = L^T*d, ghat = L^{-1}*g, Hhat = L^{-1}*H*L^{-T}
            newton step on transformed: dxNhat = -Hhat grad_hat = -L^T*H^{-1}*g = L^T*dxN
            where dxN is the original newton step
            So checking if dxNhat^T*dxNhat <= delta^2 is equivalent to dxN^T*A*dxN <= delta^2

            """
            ###### CG

            a = torch.cat([grad.view(-1) for grad in iterate.g]).data

            # Hessian_numpy=hessian.hessian.numpy()

            # cond_hessian=np.linalg.cond(Hessian_numpy)

            # print ("cond_hessian", cond_hessian)

            a1 = -1 * a
            # print ("Size of a1",a.size())
            dxN, flag_cg = conjugate_gradient_fullmat(hessian, a1, 100, 1e-6)  # 2*self._n_var

            print("dxN_size", torch.norm(dxN), "flag_cg", flag_cg)
            if flag_cg == 0:

                b = torch.cat([d.view(-1) for d in dxN]).data

                AdxN = self.Fvp(b)  # np.matmul(iterate.A,dxN)  ###### FVP

                dxN_size = torch.sqrt(torch.dot(dxN, AdxN))  ### change the variable A

                if dxN_size <= delta:
                    dx = dxN
                    dx_size = dxN_size
                    flag = "N"
                    return dx, dx_size, flag

            """
			Get the steepest descent direction taking A into account
			original QP: min g^T d + 0.5*d^T*H*d s.t. d^T A d <= delta^2
			transformed QP: min ghat^T dhat + 0.5*dhat^T*Hhat*dhat s.t. dhat^T dhat <= delta^2
			where A = L*L^T, dhat = L^T*d, ghat = L^{-1}*g, Hhat = L^{-1}*H*L^{-T}
			dxShat = -ghat
			We want to find the step size alphahat that minimizes transformed QP
			alphahat = (ghat^T*ghat)/(ghat^T*Hhat*ghat) = (g^T*A^{-1}*g)/(g^T*A^{-1}*H*A^{-1}*g)
			norm of the transformed step size is alphahat*norm(dxShat) = alphahat*sqrt(g^T*A^{-1}*g)
			If this step size is >= delta then the transformed step is -delta/(sqrt(g^T*A^{-1}*g))*ghat
			The original step -delta/(sqrt(g^T*A^{-1}*g))*Ainvg

			"""
            ##### Do CG
            # Ainvg, flag_Ag =self.conjugate_gradient(iterate.A,iterate.g,1e-6,2*self._n_var,1) #self.conjugate_gradient_solver(A,g,2*self._n_var)

            # print("size of gradient", a.size())
            # 			a1=torch.cat([grad.view(-1) for grad in iterate.g]).data
            Ainvg, flag_Ag = conjugate_gradient(self.Fvp, a, 100, 1e-6)

            # flag_Ag=flag_Ag.numpy()

            # print ("vector product",np.vdot(Ainvg,iterate.g),"flag_Ag", flag_Ag)
            alpha_hat = 0
            ghat_nrm = 0

            if flag_Ag == 0:
                # AgBAg = np.vdot(Ainvg,hessian.hessvec(Ainvg))
                AgBAg = torch.dot(Ainvg, hessian.hessvec(Ainvg))
                alpha_hat = torch.dot(Ainvg, a) / AgBAg

                ###############
                # x1=torch.dot(Ainvg,-a)*alpha_hat

                # x2=AgBAg*0.5*alpha_hat**2

                # x=-x1+x2

                # print("Decrease in function", x)
                #####################################

                ghat_nrm = torch.sqrt(torch.dot(Ainvg, a))

                dxShat_size = alpha_hat * ghat_nrm

                print("dxShat_size", dxShat_size, "delta", delta)

                if flag_cg > 0 or dxShat_size >= delta:

                    dx = -delta / ghat_nrm * Ainvg
                    dx_size = delta
                    flag = "S"
                    return dx, dx_size, flag

                if flag_cg > 0 and dxShat_size < delta:
                    dx = -alpha_hat * Ainvg
                    dx_size = dxShat_size
                    flag = "S0"
                    return dx, dx_size, flag

            ## if failed to compute the Newton Step or the steepest descent direction, resort to this

            if flag_cg > 0 or flag_Ag > 0:
                dxS = -torch.dot(a, a) / torch.dot(a, hessian.hessvec(a)) * a

                b_flat = torch.cat([d.view(-1) for d in dxS]).data
                # b_flat=b.view(-1)

                AinvdxS = self.Fvp(b_flat)

                ###----------------------------------
                dxS_size = torch.sqrt(torch.dot(dxS, AinvdxS))  #######FVPnp.matmul(iterate.A,dxS
                dx = delta / dxS_size * dxS
                dx_size = delta
                flag = "s"
                return dx, dx_size, flag

            """
			get the dogleg step taking A into account
			original QP: min g^T d + 0.5*d^T*H*d s.t. d^T A d <= delta^2
			transformed QP: min ghat^T dhat + 0.5*dhat^T*Hhat*dhat s.t. dhat^T dhat <= delta^2
			where A = L*L^T, dhat = L^T*d, ghat = L^{-1}*g, Hhat = L^{-1}*H*L^{-T}
			dxShat = -alphahat*ghat
			dxNhat = L^T*dxN
			Find alpha s.t. ||dxShat + alpha*(dxNhat - dxShat)||^2 = delta^2
			equiv. to solving ||L^T(L^{-T}*dxShat + alpha*(dxN - L^{-T}*dxShat)||^2 = delta^2
			equiv. to solving ||-alphahat*Ainvg + alpha*(dxN + alphahat*Ainvg)||^2_A = delta^2
			form the quadratic equation


			"""

            # Ainvg=np.reshape(Ainvg,[len(Ainvg),1])

            dxNAinvg = dxN + alpha_hat * Ainvg

            # print ("dxNAinvg size", dxN.shape)
            # Fvp=iterate.A
            # a=torch.from_numpy(dxNAinvg)
            # a=a.view(-1)

            # print (a.size())

            atimesdxNAing = self.Fvp(dxNAinvg)
            # atimesdxNAing=atimesdxNAing

            a_quad = torch.dot(dxNAinvg, atimesdxNAing)  ####### FVPnp.matmul(iterate.A,dxNAinvg)
            b_quad = -2 * alpha_hat * torch.dot(a, dxNAinvg)
            c_quad = alpha_hat**2 * torch.dot(Ainvg, a) - delta**2

            print("a_quad", "b_quad", "c_quad", a_quad, b_quad, c_quad)

            ## Newton step and steepest descent are parallel

            if a_quad <= 1e-6:

                dx = -delta / ghat_nrm * Ainvg
                dx_size = delta

            alpha = np.roots([a_quad, b_quad, c_quad])

            alpha_opt = np.max(alpha)

            print("alpha", alpha_opt)

            dx = -alpha_hat * Ainvg + alpha_opt * dxNAinvg

            # dx_torch=torch.from_numpy(dx)
            # dx_torch=dx_torch.view(-1)

            # Fvp=iterate.A
            Atimesdx = self.Fvp(dx)

            # Atimesdxtorch=Atimesdxtorch.numpy()
            dx_size = torch.sqrt(torch.dot(dx, Atimesdx))  ######## FVP np.matmul(iterate.A,dx)

            if alpha_opt < 0:
                print("alpha_opt", alpha_opt)

                logging.debug("Error in computing the dogleg step")

                dx = []

                dx_size = 0

            if abs(dx_size - delta) >= 1e-2:

                print("distance", abs(dx_size - delta))
                logging.debug("Error in computing Dogleg Step")

            flag = "D"
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
