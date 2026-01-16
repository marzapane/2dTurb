#!/opt/intel/oneapi/intelpython/latest/bin/python

import argparse
from po2d_config import *
from po2d_lib import *
import numpy as np
import scipy.sparse.linalg as sla

def vertical_flip(square: np.ndarray):
    n = square.shape[0]
    return square[:, n-1::-1]

def horizontal_flip(square: np.ndarray):
    n = square.shape[0]
    return square[n-1::-1, :]

def main_diagonal_flip(square: np.ndarray):
    n = square.shape[0]
    return square[n-1::-1, n-1::-1].transpose()    

def secondary_diagonal_flip(square: np.ndarray):
    return square.transpose()

def clockwise_rotation(square: np.ndarray):
    n = square.shape[0]
    return square.transpose()[:, n-1::-1]

def counter_clockwise_rotation(square: np.ndarray):
    n = square.shape[0]
    return square.transpose()[n-1::-1, :]

def pi_rotation(square: np.ndarray):
    n = square.shape[0]
    return square[n-1::-1, n-1::-1]

def print_symmetries(eig_system, h):
    print(' i    <qh>    m^90  m^/  m^\\  m^|  m^-   #')
    for i in range(64):
        eig_fn = eig_system[i+1]
        norm = np.square(eig_fn).sum()
        print(f'{i+1:02}', end='  ')
        z_sum = (eig_fn*h).sum()
        print(f'{z_sum:+.1e}', end='  ')
        ash = 0
        m = (eig_fn * counter_clockwise_rotation(eig_fn)).sum() / norm
        print(' X  ', end='  ') if np.isclose(m, +1, atol=1e-2) else print('    ', end='  ')
        if np.isclose(m, +1, atol=1e-2): ash += 1
        m = (eig_fn * main_diagonal_flip(eig_fn)).sum() / norm
        print(' X ', end='  ') if np.isclose(m, -1, atol=1e-2) else print('   ', end='  ')
        if np.isclose(m, -1, atol=1e-2): ash += 2
        m = (eig_fn * secondary_diagonal_flip(eig_fn)).sum() / norm
        print(' X ', end='  ') if np.isclose(m, -1, atol=1e-2) else print('   ', end='  ')
        if np.isclose(m, -1, atol=1e-2): ash += 4
        m = (eig_fn * horizontal_flip(eig_fn)).sum() / norm
        print(' X ', end='  ') if np.isclose(m, -1, atol=1e-2) else print('   ', end='  ')
        if np.isclose(m, -1, atol=1e-2): ash += 8
        m = (eig_fn * vertical_flip(eig_fn)).sum() / norm
        print(' X ', end='  ') if np.isclose(m, -1, atol=1e-2) else print('   ', end='  ')
        if np.isclose(m, -1, atol=1e-2): ash += 16
        print(f'{ash:02}')

def refine_eigenpair(
    A,
    eig,
    v,
    tol=1e-10,
    maxiter=None,
):
    """
    Refine an approximate eigenpair (lam, v) of a sparse matrix A.
    Works for general (possibly non-symmetric) matrices.
    """
    from scipy.sparse import eye
    n = A.shape[0]
    def matvec(x):
        return A @ x - eig * x
    residual = np.linalg.norm(matvec(v))
    w = sla.spsolve(A - eig * eye(A.shape[0], dtype=A.dtype), v)
    v_new = w / np.linalg.norm(w)
    eig_new = np.vdot(v_new, A @ v_new)
    residual_new = np.linalg.norm(A @ v_new - eig_new * v_new)
    if residual_new < residual:
        return eig_new, v_new, residual_new
    else:
        print('Result got worse. Returning old values')
        return eig, v, residual

def main(symmetrize):
    # setup q[] = 1/h^2 [1/h grad(h).grad - Lap]
    topo = gauss_topography(Dx, N)
    h = 1 + 2/3 * topo/topo.max()
    dx_h = derivative(h, 0, Dx)
    dy_h = derivative(h, 1, Dx)
    H1 = (dx_h / np.power(h, 3)).flatten()
    H2 = (dy_h / np.power(h, 3)).flatten()
    with np.errstate(divide="ignore"):
        Ih = 1 / np.power(h, 2).flatten()
    dx = dx_spmat(Dx, N)
    dy = dy_spmat(Dx, N)
    lap = laplacian_spmat(Dx, N)
    q_of_psi = dx.multiply(H1[:, None]) + dy.multiply(H2[:, None]) - lap.multiply(Ih[:, None])

    # compute eigenvalues and eigenvectors
    eigs, eigv = sla.eigs(q_of_psi, min(N//2+1, 65), which='SM', tol=0)

    # sort and store them
    sort_idx = np.argsort(eigs)
    eigs = eigs[sort_idx]
    eigv = eigv[:, sort_idx].T
    lst_eigv = []
    for f in eigv:
        lst_eigv.append(f.reshape(N, N))
    eigv = np.asarray(lst_eigv).real

    if (symmetrize):
        # symmetrize states in larger than 1d eigenspaces
        tol = 1e-10
        i = 0
        while i < len(eigs):
            deg_eig = [i]
            j = i + 1
            while j < len(eigs) and abs(eigs[j] - eigs[i]) < tol:
                deg_eig.append(j)
                j += 1
            if len(deg_eig) > 1:
                print(f"Degenerate group found for eigenvalue {eigs[i]:.6f}: Indices {deg_eig}")
                if len(deg_eig) == 2:
                    qsym = (eigv[i] + eigv[i].T)/2
                    # eig, qsym, res = refine_eigenpair(q_of_psi, eigs[i], qsym.flatten())
                    eigv[i+1] = qsym.reshape(N,N)
                    qsym = (eigv[i] - eigv[i].T)/2
                    # eig, qsym, res = refine_eigenpair(q_of_psi, eigs[i], qsym.flatten())
                    eigv[i] = qsym.reshape(N,N)
                else:
                    print("Eigenspace too large for symmetries")
            i = j
        
    # fix zero sum for rotationally symmetric states
    for i in range(1, len(eigs)):
        if np.isclose(eigv[i], clockwise_rotation(eigv[i])).all():
            eigv[i] -= (eigv[i]*h).sum()/h.sum()

    # rescale energy to 1
    for i in range(1, len(eigv)):
        fluid = FluidStateTopography(pot_vorticity=eigv[i], topography=topo)
        fluid.streamfunction()
        E = fluid.energy()
        eigv[i] = eigv[i] / np.sqrt(E)

    # export
    np.savez('../results/eigfn.npz', eigs=eigs, eigfn=eigv)
    
    print_symmetries(eigv, h)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute lowest eigenstates of q(psi).")
    parser.add_argument("--sym", action="store_true", default=False, help="Symmetrize eigenstates")
    main(parser.parse_args().sym)