"""Weak tangent projection of the curve, its exact time derivative and the surrogate acceleration energy E|a_M|^2.

Let F(x) = [grad f_1(x) ... grad f_M(x)] in R^{d x M} be the gradients of a dictionary of potentials and (X, U, W)
positions, velocities and accelerations of the generator's particles at one time. With E the mean over a batch:

    A    = E[F^T F],                    b    = E[F^T U],                    (A + rI) c = b,       v_M = F c,
    Fdot = D_x F[U],                    Adot = E[Fdot^T F + F^T Fdot],      bdot = E[Fdot^T U + F^T W],
    (A + rI) cdot = bdot - Adot c,      a_M  = F cdot + D_x F[F c] c.

c(t) are the coefficients of the ridge-regularised L2(rho_t) projection of the particle velocity onto the gradient
fields of the dictionary and cdot their exact time derivative (differentiating (A + rI) c = b); the term Adot c is
the derivative of the Gram matrix along the moving measure (moving-Gram term). a_M is the material derivative
d_t v_M + D_x v_M v_M of the projected velocity. For a fixed dictionary and ridge the population moments A, b, Adot,
bdot (the expectations that these batch means estimate) depend on the marginal curve only (continuity equation:
d/dt E[g(X_t)] = E[grad g(X_t) . U_t]), so the population energy E|a_M|^2 does not depend on how particles are paired
(its minibatch estimate only up to sampling error). Nothing is detached: gradients flow through the moments, both
Cholesky solves and both batches. The projection batch and the batch on which the energy is evaluated are independent.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .model import SplineGenerator
from .potentials import Dictionary


def assemble(F: Tensor, U: Tensor, Fdot: Tensor, W: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Moments ``(A, b, Adot, bdot)`` of the projection with Monte-Carlo weights ``w_i = 1/B``.

    ``A = sum_i w_i F_i^T F_i`` (symmetrised), ``b = sum_i w_i F_i^T U_i``, ``Adot = C + C^T`` with
    ``C = sum_i w_i Fdot_i^T F_i``, ``bdot = sum_i w_i (Fdot_i^T U_i + F_i^T W_i)``.

    Args:
        F, Fdot: (B, d, M) gradients and their derivative ``D_x F[U]`` at the samples.
        U, W: (B, d) particle velocities and accelerations.
    """
    w = torch.full((F.shape[0],), 1.0 / F.shape[0], dtype=F.dtype, device=F.device)
    Fw = F * w.view(-1, 1, 1)
    A = torch.einsum("bdm,bdn->mn", Fw, F)
    A = 0.5 * (A + A.transpose(0, 1))
    b = torch.einsum("bdm,bd->m", Fw, U)
    # the weighted Fdot is formed twice on purpose: the operation order fixes the bits of the result
    cross = torch.einsum("bdm,bdn->mn", Fdot * w.view(-1, 1, 1), F)
    Adot = cross + cross.transpose(0, 1)
    bdot = (torch.einsum("bdm,bd->m", Fdot * w.view(-1, 1, 1), U)
            + torch.einsum("bdm,bd->m", Fw, W))
    return A, b, Adot, bdot


def solve_projection(F: Tensor, U: Tensor, Fdot: Tensor, W: Tensor, ridge: float) -> tuple[Tensor, Tensor]:
    """Coefficients ``c`` of the projected velocity and their time derivative ``cdot`` (each (M,)).

    Solves ``(A + rI) c = b`` and ``(A + rI) cdot = bdot - Adot c`` with one Cholesky factor of ``A + rI``.

    Raises:
        torch.linalg.LinAlgError: if ``A + rI`` is not positive definite.
    """
    A, b, Adot, bdot = assemble(F, U, Fdot, W)
    eye = torch.eye(A.shape[0], dtype=A.dtype, device=A.device)
    L = torch.linalg.cholesky(A + ridge * eye)
    c = torch.cholesky_solve(b.unsqueeze(1), L).squeeze(1)
    cdot = torch.cholesky_solve((bdot - Adot @ c).unsqueeze(1), L).squeeze(1)
    return c, cdot


def acceleration(dic: Dictionary, X: Tensor, c: Tensor, cdot: Tensor) -> Tensor:
    """Projected acceleration ``a_M(x) = F(x) cdot + D_x F(x)[F(x) c] c`` at X (B, d), shape (B, d).

    The second term is ``D_x v_M[v_M]``: the Hessians of the potentials applied to ``v_M = F c`` and contracted with
    c (closed-form Hessian-vector products).
    """
    F = dic.grad(X)
    v = torch.einsum("bdm,m->bd", F, c)
    term1 = torch.einsum("bdm,m->bd", F, cdot)
    term2 = torch.einsum("bdm,m->bd", dic.hessvec(X, v), c)
    return term1 + term2


def intrinsic_energy(gen: SplineGenerator, dic: Dictionary, t: float, n_proj: int, n_loss: int, ridge: float,
                     generator: torch.Generator | None = None) -> Tensor:
    """``E|a_M(t, X_t)|^2`` at normalised time t (a 0-dim tensor, differentiable in the generator's parameters).

    Draws from ``generator`` in this order: a projection batch of ``n_proj`` latents (X, U, W), used to solve the
    projection, then an independent loss batch of ``n_loss`` latents (X), on which the energy is averaged.
    """
    zp = gen.sample_latents(n_proj, generator=generator)
    Xp, Up, Wp = gen.evaluate([t], zp, orders=(0, 1, 2))
    Xp, Up, Wp = Xp[0], Up[0], Wp[0]
    c, cdot = solve_projection(dic.grad(Xp), Up, dic.hessvec(Xp, Up), Wp, ridge)
    zl = gen.sample_latents(n_loss, generator=generator)
    Xl = gen.evaluate([t], zl, orders=(0,))[0][0]
    a = acceleration(dic, Xl, c, cdot)
    return (a ** 2).sum(1).mean()
