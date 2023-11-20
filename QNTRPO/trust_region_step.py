# Copyright (C) 2019, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Python script to compute the trust region step """

# from conjugate_grad_solver import conjugate_gradients
# import scipy.sparse
import logging

import numpy as np
import torch

logging.basicConfig(level=logging.DEBUG)

from conjugate_grad_solution import conjugate_gradient

logger = logging.getLogger(__name__)


class trpo_step(object):
    def __init__(self, n_var, type_option, model, get_loss, get_kl, damping):

        self._n_var = n_var
        self.type = type_option

        self.model = model
        self.get_kl = get_kl
        self.get_loss = get_loss
        self.damping = damping

    # 		self.conjugate_gradient_solver=conjugate_gradients

    def Fvp(self, v):
        get_kl = self.get_kl
        loss = self.get_loss()
        damping = self.damping
        model = self.model

        kl = get_kl()
        kl = kl.mean()

        grads = torch.autograd.grad(kl, model.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_v = (flat_grad_kl * Variable(v)).sum()

        grads = torch.autograd.grad(kl_v, model.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data

        return flat_grad_grad_kl + v * damping

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
            dxN, flag_cg = self.conjugate_gradient(hessian, -iterate.g, 1e-6, 2 * self._n_var, 0)

            print("flag_cg", flag_cg)
            if flag_cg == 0:

                a = torch.from_numpy(dxN)
                a = a.view(-1)
                Fvp = iterate.A
                AdxN = Fvp(a)  # np.matmul(iterate.A,dxN)  ###### FVP
                AdxN = AdxN.numpy()
                dxN_size = np.sqrt(np.vdot(dxN, AdxN))  ### change the variable A

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

            a = torch.from_numpy(iterate.g)
            a = a.view(-1)

            # print("size of gradient", a.size())
            Ainvg, flag_Ag = conjugate_gradient(self.Fvp, a, 2 * self._n_var, 1e-6)

            Ainvg = Ainvg.numpy()

            print("Ainvg size", Ainvg.shape)
            # flag_Ag=flag_Ag.numpy()

            print("vector product", np.vdot(Ainvg, iterate.g), "flag_Ag", flag_Ag)
            alpha_hat = 0
            ghat_nrm = 0

            if flag_Ag == 0:
                AgBAg = np.vdot(Ainvg, hessian.hessvec(Ainvg))
                alpha_hat = np.vdot(Ainvg, iterate.g) / AgBAg

                ###############
                x1 = np.vdot(Ainvg, iterate.g) * alpha_hat

                x2 = AgBAg * 0.5 * alpha_hat**2

                x = -x1 + x2

                print("Decrease in function", x)
                #####################################

                ghat_nrm = np.sqrt(np.vdot(Ainvg, iterate.g))

                dxShat_size = alpha_hat * ghat_nrm

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
                dxS = -np.vdot(iterate.g, iterate.g) / np.vdot(iterate.g, hessian.hessvec(iterate.g)) * iterate.g
                Fvp = iterate.A
                b = torch.from_numpy(dxS)
                b = b.view(-1)

                AinvdxS = Fvp(b)
                AinvdxS = AinvdxS.numpy()

                ###----------------------------------
                dxS_size = np.sqrt(np.vdot(dxS, AinvdxS))  #######FVPnp.matmul(iterate.A,dxS
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

            Ainvg = np.reshape(Ainvg, [len(Ainvg), 1])

            dxNAinvg = dxN + alpha_hat * Ainvg

            print("dxNAinvg size", dxN.shape)
            Fvp = iterate.A
            a = torch.from_numpy(dxNAinvg)
            a = a.view(-1)

            print(a.size())

            atimesdxNAing = Fvp(a)
            atimesdxNAing = atimesdxNAing.numpy()

            a_quad = np.vdot(dxNAinvg, atimesdxNAing)  ####### FVPnp.matmul(iterate.A,dxNAinvg)
            b_quad = -2 * alpha_hat * np.vdot(iterate.g, dxNAinvg)
            c_quad = alpha_hat**2 * np.vdot(Ainvg, iterate.g) - delta**2

            ## Newton step and steepest descent are parallel

            if a_quad <= 1e-6:

                dx = -delta / ghat_nrm * Ainvg
                dx_size = delta

            alpha = np.roots([a_quad, b_quad, c_quad])

            alpha_opt = np.max(alpha)

            dx = -alpha_hat * Ainvg + alpha_opt * dxNAinvg

            dx_torch = torch.from_numpy(dx)
            dx_torch = dx_torch.view(-1)

            Fvp = iterate.A
            Atimesdxtorch = Fvp(dx_torch)

            Atimesdxtorch = Atimesdxtorch.numpy()
            dx_size = np.sqrt(np.vdot(dx, Atimesdxtorch))  ######## FVP np.matmul(iterate.A,dx)

            if alpha_opt < 0:

                logging.debug("Error in computing the dogleg step")

                dx = []

                dx_size = 0

            if abs(dx_size - delta) >= 1e-2:
                logging.debug("Error in computing Dogleg Step")

            flag = "D"
            return dx, dx_size, flag

        if self.type == 0:

            ## Compute scaled steepest descent #### DO CG
            # [Ainvg,flag_Ag] = self.conjugate_gradient(iterate.A,iterate.g,1e-6,2*self._n_var,1)
            a = torch.from_numpy(iterate.g)
            a = a.view(-1)

            Ainvg, flag_Ag = conjugate_gradient(self.Fvp, -a, 2 * self._n_var, 1e-6)
            # Ainvg=Ainvg.numpy()
            # flag_Ag=flag_Ag.numpy()

            ## do linsearch
            vecpdt = torch.dot(Ainvg, -a)

            success, new_params = linesearch(model, get_loss, prev_params, fullstep, vecpdt)

            print("flag_Ag", flag_Ag)

            if flag_Ag == 0:
                print("In the scaling loop")

                vecpdt = torch.dot(Ainvg, -a)
                vecpdt = vecpdt.numpy()
                scaledx = delta / np.sqrt(vecpdt)

                print("Scale dx", scaledx)
                dx = 1 * Ainvg.numpy()  # scaledx
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

        if self.type == 2:

            ## Compute combined step
            [dx, flag, flag_step] = self.conjugate_gradient_steihaug(hessian, iterate.g, 1e-6, 2 * self._n_var, delta)

            dx_size = np.linalg.norm(dx)
            flag = "C"

            return dx, dx_size, flag_step

    def conjugate_gradient(self, hessOrMat, b, residual_tol, nsteps, flag):

        ## flag == 0: hessian, 1: matrix
        x = np.zeros([len(b), 1])
        nrmb = np.linalg.norm(b)
        if flag == 0:
            r = -b + hessOrMat.hessvec(x)
        else:
            r = -b + np.matmul(hessOrMat, x)
        p = -np.copy(r)
        rdotr = np.vdot(r, r)  # torch.dot(r,r)
        # 		print("Initial residual = ",np.linalg.norm(r))
        flag_cg = 1
        if np.sqrt(rdotr) <= residual_tol or nrmb <= residual_tol or np.sqrt(rdotr) / nrmb <= residual_tol:
            flag_cg = 0
            return x, flag_cg

        for i in range(nsteps):
            if flag == 0:
                _Avp = hessOrMat.hessvec(p)
            else:
                _Avp = np.matmul(hessOrMat, p)
            p_Avp = np.vdot(p, _Avp)
            alpha = rdotr / (p_Avp)  # torch.dot(p,_Avp)
            x += alpha * p
            r += alpha * _Avp

            new_rdotr = np.vdot(r, r)  # torch.dot(r,r)

            beta = new_rdotr / rdotr

            p = -r + beta * p

            rdotr = new_rdotr

            if np.sqrt(rdotr) <= residual_tol or np.sqrt(rdotr) / nrmb <= residual_tol:
                flag_cg = 0
                break
        # 		print("CG: ",rdotr," its ",i)

        return x, flag_cg

    def conjugate_gradient_steihaug(self, hessOrMat, b, residual_tol, nsteps, delta):

        ## flag == 0: hessian, 1: matrix
        z = np.zeros([len(b), 1])
        r = np.copy(b)
        d = -np.copy(r)
        nrmb = np.linalg.norm(b)
        rdotr = np.vdot(r, r)  # torch.dot(r,r)
        # 		print("Initial residual = ",np.linalg.norm(r))
        flag_cg = 1
        if np.sqrt(rdotr) <= residual_tol or nrmb <= residual_tol or np.sqrt(rdotr) / nrmb <= residual_tol:
            flag_cg = 0
            flag_step = "S"
            return z, flag_cg, flag_step

        for i in range(nsteps):
            _Avd = hessOrMat.hessvec(d)
            d_Avd = np.vdot(d, _Avd)
            alpha = rdotr / d_Avd  # torch.dot(p,_Avp)
            z1 = z + alpha * d
            if np.linalg.norm(z1) >= delta:
                a_quad = np.vdot(d, d)
                b_quad = 2 * np.vdot(d, z)
                c_quad = np.vdot(z, z) - delta**2
                alpha = np.roots([a_quad, b_quad, c_quad])
                alpha_opt = np.max(alpha)
                z = z + alpha_opt * d
                flag_cg = 0
                flag_step = "C"
                # 				logging.debug("qCG: %e its %d x1 %e",rdotr,i,np.linalg.norm(z1))
                return z, flag_cg, flag_step

            r += alpha * _Avd

            new_rdotr = np.vdot(r, r)  # torch.dot(r,r)

            beta = new_rdotr / rdotr

            d = -r + beta * d

            rdotr = new_rdotr
            z = z1

            if np.sqrt(rdotr) <= residual_tol or np.sqrt(rdotr) / nrmb <= residual_tol:
                flag_step = "N"
                flag_cg = 0
                break
        # 		logging.debug("CG: %e its %d res %e",rdotr,i,np.linalg.norm(hessOrMat.hessvec(z)-b))

        return z, flag_cg, flag_step
