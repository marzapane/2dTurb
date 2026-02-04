import numpy as np
import numpy.fft as ft
import numpy.linalg as la
import scipy.special as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import SymLogNorm, Normalize
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.stats import binned_statistic
from scipy.ndimage import gaussian_filter
from pathlib import Path
try:
    from functools import cache
except ImportError:
    # Fallback: Define a decorator that does nothing
    def cache(func):
        return func

class FluidSimulator:
    from po2d_config import T, N, Re, ekman, eps, cfg_name, sd_len, Dx, Dt
    reload_bak = True
    
    gamma = np.array((8/15, 5/12, 3/4))
    rho = np.array((0, -17/60, -5/12))
    alpha = gamma + rho
    # L = [None, None, None]
    eigs_M = [None, None, None]

    def __init__(
        self,
        forcing: callable,
        adaptive_Dt = False,
        plot_flag = True,
        analize_vortex = False,
    ):
        self.forcing = forcing
        self.analize_vortex = analize_vortex
        self.adapt_Dt = adaptive_Dt
        self.diagnostics = False
        self.plot_flag = plot_flag
        self.silent = False
        if plot_flag: self.plotter = VorticityPlotter(self.Dx, self.N)


    def initialize_run_state(self):
        """Initializes folders, checks for previous simulations, and sets up time/logging."""
        self.diagnostics = True
        print(f'Running simulation "{self.cfg_name}"')
        self.frames_dir, self.bak_dir = self.setup_folders() # looking for past simulations backup files
        self.bak_file = find_q(self.bak_dir)
        self.reload_bak = self.confirm_reload()
        open_folder(self.bak_dir, overwrite = not self.reload_bak)
        open_folder(self.frames_dir, overwrite = not self.reload_bak)
        if self.reload_bak and self.bak_file:
            self.t0 = int(self.bak_file.stem[1:])
            self.time = float(np.load(self.bak_file)['t'])
            try:
                self.N      = int(np.load(self.bak_file)['N'])
                self.Re     = float(np.load(self.bak_file)['Re'])
                self.ekman  = float(np.load(self.bak_file)['ek'])
                self.eps    = float(np.load(self.bak_file)['eps'])
                self.sd_len = float(np.load(self.bak_file)['sd'])
                self.hb_scl = float(np.load(self.bak_file)['hb'])
            except KeyError: pass
                # print("No simulation parameter found in backup file, using po2d_config's values.")
            self.stat_file = open(self.bak_dir / 'time_stat.dat', 'a')
        else:
            self.t0 = 0
            self.time = 0.
            self.stat_file = open(self.bak_dir / 'time_stat.dat', 'w')
            print('# time E eps avg_k q2_sum q_sum c(q)', file=self.stat_file)
        if self.analize_vortex:
            max_dist = int(self.N/2 * np.sqrt(2))    # furthest distance from the domain center
            self.bkgnd_sum = np.zeros(max_dist)
            self.bins = np.arange(max_dist+1) * self.Dx

    def setup_folders(self):
        result_dir = Path().resolve().parent / 'results'
        if not result_dir.exists():
            result_dir.mkdir()
        fold_name = f'Ek{self.ekman:.1e},Re{self.Re:.0e},N{self.N}'
        if self.cfg_name != '':
            fold_name = f'{self.cfg_name},{fold_name}'
        bak_dir = result_dir / fold_name
        frames_dir = bak_dir / f'frames'
        return frames_dir, bak_dir

    def confirm_reload(self):   # returns the answer to "Restart past simulation?"
        if self.bak_file:   # asking user confirmation to reload last simulation (default)
            print(f'A previous simulation has been found: {self.bak_file.name}.')
            if self.silent:
                print(f'Restarting past simulation from t0={self.bak_file.stem[1:]}.')
                return True
            else:
                print('Starting a different one will overwrite it.')
                user_input = input('  Reload last simulation? [Y/n] ')
                if user_input.lower() in ['no', 'n']:
                    print('Launching a new simulation and overwriting the past one.')
                    return False
                else:
                    print(f'Restarting past simulation from t0={self.bak_file.stem[1:]}.')
                    return True
        else:
            print('No previous simulation was found; launching a new one.')
            return False

    def set_physical_param(self, fluid):
        self.set_integration_const()
        if self.adapt_Dt:
            self.max_CFL = np.sqrt(3)  # for 3rd order Runge-Kutta
            self.cur_CFL = 1
        self.T_print = self.time
        if fluid.q.any():
            E = fluid.energy()
            eps = fluid.energy_dissipation(self.Re)
            # E_in = fluid.energy_input(self.F)
            S = energy_spectrum(*fluid.velocity())
            avg_k = np.average(np.arange(S.size), weights=S) if S.any() else 0.0
            eddy_turnover = (2*np.pi/avg_k) / np.sqrt(2*E)
            self.dT_print = min(10*eddy_turnover, self.T//5)
            # revol_time = pow(self.sd_len**2/(2*eps), 1/3)
            # self.dT_print = min(int(revol_time/self.Dt), self.T//5)
        else:
            self.dT_print = self.T//5
        self.T_update = 10
        # print(f'Simulation times are:\n  Dt = {self.Dt}\n  T_LE ~ {revol_time}')
        if self.silent:
            self.time_exec = np.arange(self.T+1)
        else:
            self.time_exec = tqdm(np.arange(self.T+1), dynamic_ncols=True)
        self.F = self.forcing(self.N, self.eps)
        fluid.streamfunction()
        fluid.arakawa_jacobian()

    def set_integration_const(self):
        self.c = self.Dt * self.alpha / (self.Re * self.Dx**2)
        self.d = self.Dt * self.alpha * self.ekman
        n = np.arange(self.N)
        cos = np.cos(2*np.pi * n/self.N)
        for i in range(3):
            self.eigs_M[i] = (1 + 2*self.c[i] + self.d[i]/2) - self.c[i]*(cos[:, None] + cos[None, :self.N//2+1])
            # self.L[i] = self.linear_sys_inv(self.c[i], self.d[i], self.N)

    # @staticmethod
    # def linear_sys_inv(
    #     c: float,   # Runge-kutta coefficients
    #     d: float,
    #     n: int,     # grid size
    # ):
    #     M = (1 + c + d/2) * np.eye(n) - np.diag(c/2 * np.ones(n-1), 1) - np.diag(c/2 * np.ones(n-1), -1)
    #     M[0, n-1] = M[n-1, 0] = -c/2
    #     return la.inv(M)
    
    def advance_dt(
        self,
        fluid,
        t: int,
    ):
        for step in range(3):
            self.rungekutta_step(fluid, step)
        self.time += self.Dt
        fluid.qsum20()
        if (t % self.T_update) == 0:
            E = fluid.energy()
            eps = fluid.energy_dissipation(self.Re)
            # E_in = fluid.energy_input(self.F)
            S = energy_spectrum(*fluid.velocity())
            avg_k = np.average(np.arange(S.size), weights=S) if S.any() else 0.0
            if np.isfinite(self.Re):
                revol_time = pow(self.sd_len**2/(2*eps), 1/3)
                eddy_turnover = (2*np.pi/avg_k) / np.sqrt(2*E)
            else:
                revol_time = 4*np.pi / np.abs(fluid.vorticity()).mean()
            # self.dT_print = min(int(revol_time/self.Dt), self.T//5)
            seldom = max(200, 1600 * np.log10(self.T) - 4600) if (self.T < 1e6) else 5e3
            self.dT_print = min(10*eddy_turnover, seldom*self.Dt)
            if self.silent:
                print(f't = {self.time:.1f} | E = {E:.2g} | <k> = {avg_k:.2g} | eps = {eps:.2g}')
            else:
                self.time_exec.set_description(f't = {self.time:.1f} | E = {E:.2g} | <k> = {avg_k:.2g} | eps = {eps:.2g}')
            if self.diagnostics:
                print(self.time, E, eps, avg_k, fluid.enstrophy(), fluid.casimir(1), measure_concentration(fluid.q, self.N), sep='\t', file=self.stat_file, flush=True)
            if self.adapt_Dt:
                fluid.velocity()
                max_velocity = np.hypot(fluid.v_x, fluid.v_y).max()
                line_Dt = (self.cur_CFL * self.Dx / max_velocity / 10) if (max_velocity > 0) else self.Dt
                if np.isfinite(self.Re):
                    turb_Dt = revol_time/np.sqrt(self.Re)
                    Dt = min(line_Dt, turb_Dt, 50) / 4
                else:
                    Dt = line_Dt
                if abs(Dt - self.Dt) > self.Dt / 5:
                    self.Dt = Dt
                    # print(f'    !!  v({t})<{max_velocity:.3g}  &  T_rev={revol_time:.1f}   ->   Dt = {self.Dt:.2g}')
                    self.set_integration_const()
        if self.time >= self.T_print + self.dT_print:
            self.T_print = self.time
            if self.diagnostics:
                np.savez(self.bak_dir / f'q{(t+self.t0):08}',
                         q  = fluid.q,
                         t  = self.time,
                         N  = self.N,
                         Re = self.Re,
                         ek = self.ekman,
                         eps= self.eps,
                         sd = self.sd_len,
                         f  = fluid.f_Coriolis,
                         hb = fluid.hb_scl
                        )
            if self.plot_flag:
                fig_path = self.frames_dir / f'{(t+self.t0):08}.png'
                self.plotter.update(fluid.q, self.time, savepath=fig_path)
                # fluid.plot_field(self, self.time, savepath=fig_path)
        if self.analize_vortex:
            fluid.velocity()
            self.bkgnd_sum = self.bkgnd_sum + fluid.avg_centered_field(self.bins) + (-fluid).avg_centered_field(self.bins)

    # def rungekutta_step_nofft(
    #     self,
    #     fluid,
    #     step: int,
    # ):
    #     F_p = self.F.copy()
    #     self.F = self.forcing(self.N, self.eps)
    #     J_p = fluid.J.copy()
    #     fluid.arakawa_jacobian()
    #     rhs = self.Dt* (self.gamma[step]*(self.F - fluid.J) + self.rho[step]*(F_p - J_p)) - self.d[step]*fluid.q + self.c[step]*fluid.dissipation()
    #     Dq = np.matmul(np.matmul(self.L[step], rhs), self.L[step].transpose())
    #     fluid.q += Dq
    #     self.upd_psi = True
    #     self.upd_J = True
    #     self.upd_v = True

    def rungekutta_step(
        self,
        fluid,
        step: int,
    ):
        F_p = self.F.copy()
        self.F = self.forcing(self.N, trg_eps=self.eps, Dt=self.Dt)
        J_p = fluid.J.copy()
        fluid.arakawa_jacobian()
        rhs = self.Dt* (self.gamma[step]*(self.F - fluid.J) + self.rho[step]*(F_p - J_p)) - self.d[step]*fluid.q + self.c[step]*fluid.dissipation()
        fluid.q += self.inv_M_fft(rhs, step)
        self.upd_psi = True
        self.upd_J = True
        self.upd_v = True
    
    def inv_M_fft(
        self,
        rhs,
        i: int,
    ):
        rhs_ft = ft.rfft2(rhs)
        inv_ft = ft.rfft2(rhs) / self.eigs_M[i]
        return ft.irfft2(inv_ft, s=rhs.shape)

    def conclude(self):
        if self.diagnostics and not self.stat_file.closed:
            self.stat_file.close()
        if self.analize_vortex:
            self.bkgnd = self.bkgnd_sum / (2 * (self.T + 1))
            np.save(self.bak_dir / f'bkgnd', np.array((self.bins[:-1], self.bkgnd)))
        if self.plot_flag: self.plotter.close()
    

class FluidState:
    from po2d_config import N, Dx, f_Coriolis, hb_scl

    def __init__(
        self,
        simul = None,
        pot_vorticity = None,
        rel_vorticity = None,
        f_Coriolis = None,
    ):
        if f_Coriolis is not None:
            self.f_Coriolis = f_Coriolis
        if simul is None:
            self.init_vorticity(pot_vorticity, rel_vorticity)
        else:
            if simul.reload_bak:
                self.q = np.load(simul.bak_file)['q']
                try:
                    self.f_Coriolis = np.load(simul.bak_file)['f']
                except KeyError:
                    print("No simulation parameter found in backup file, using po2d_config's values.")
            else:
                self.init_vorticity(pot_vorticity, rel_vorticity)
                if simul.diagnostics:
                    np.savez(
                             simul.bak_dir / f'q{0:08}',
                             q  = self.q,
                             t  = 0.0,
                             N  = simul.N,
                             Re = simul.Re,
                             ek = simul.ekman,
                             eps= simul.eps,
                             sd = simul.sd_len,
                             f  = self.f_Coriolis,
                             hb = self.hb_scl
                            )
                if simul.plot_flag:
                    fig_path = simul.frames_dir / f'{0:08}.png'
                    simul.plotter.update(self.q, 0.0, savepath=fig_path)

        self.psi = None
        self.upd_psi = True
        self.J = None
        self.upd_J = True
        (self.v_x, self.v_y) = (None, None)
        self.upd_v = True

    def init_vorticity(
        self,
        pot_vorticity,
        rel_vorticity,
    ):
        if rel_vorticity is not None:
            q = rel_vorticity + self.f_Coriolis
            if (pot_vorticity is not None):
                raise Exception("Error: wrong arguments: cannot set both potential and relative vorticity.")
        else:
            q = pot_vorticity

        if q is not None:
            if isinstance(q, np.ndarray) and q.shape == (self.N, self.N):
                self.q = q
            else:
                raise Exception("Error: wrong argument: q must be <numpy.ndarray> of shape (N, N).")
        else:
            self.q = self.f_Coriolis

    def streamfunction(
        self,
    ):
        if self.upd_psi:
            self.psi = - inv_laplacian2d(self.q - self.f_Coriolis, self.Dx)
            self.upd_psi = False
        return self.psi

    def velocity(
        self,
    ):
        if self.upd_v:
            self.streamfunction()
            self.v_x = derivative(self.psi, 1, self.Dx)  # d(psi)/dy
            self.v_y = -derivative(self.psi, 0, self.Dx) # -d(psi)/dx
            self.upd_v = False
        return self.v_x, self.v_y
    
    def vorticity(self):
        return self.q - self.f_Coriolis

    def dissipation(self):
        return pseudo_laplacian2d(self.vorticity())

    def energy(self):
        return integrate(self.vorticity() * self.streamfunction(), self.Dx) / 2

    def casimir(
        self,
        pow: int,
    ):
        return integrate(np.power(self.q, pow), self.Dx) / pow

    def enstrophy(self):
        return self.casimir(2)

    def energy_dissipation(
        self,
        Re: float,
    ):
        (dudx, dvdx) = [derivative(v, 0, self.Dx) for v in self.velocity()]
        (dudy, dvdy) = [derivative(v, 1, self.Dx) for v in self.velocity()]
        return integrate((dudx**2 + dvdx**2 + dudy**2 + dvdy**2), self.Dx) / Re

    def energy_input(
        self,
        F: np.ndarray,
    ):
        return integrate(F * self.streamfunction(), self.Dx)
    
    def avg_centered_field(
        self,
        bins: np.ndarray,
    ):
        vortex_ctr = self.find_vortex_center()
        dist, angle = relative_pos(*vortex_ctr, self.Dx, self.N)
        omega = self.avg_vorticity(dist, angle)
        avg_omega, _, _ = binned_statistic(dist.flatten(), omega.flatten(), statistic='mean', bins=bins)
        return avg_omega

    def find_vortex_center(self):
        ctr = self.N//2     # grid center
        hr = self.N//16     # half range of the interval considered
        q_flt = gaussian_filter(self.q, sigma=self.N/25, truncate=2., mode='wrap') # gaussian convolution, to exclude large fluctuations
        (x_max, y_max) = np.unravel_index(q_flt.argmax(), q_flt.shape)
        max_ctr_v_x = np.roll(np.roll(self.v_x, ctr-x_max, axis=0), ctr-y_max, axis=1)  #bring the max in the position (N/2, N/2)
        max_ctr_v_y = np.roll(np.roll(self.v_y, ctr-x_max, axis=0), ctr-y_max, axis=1)
        zero_max_ctr_v = zero_coord(max_ctr_v_x, max_ctr_v_y, ctr-hr, ctr+hr)
        grid_ctr = zero_max_ctr_v + np.array([x_max, y_max])-ctr    # revert reference frame change
        return (grid_ctr % self.N) * self.Dx

    def avg_vorticity(
        self,
        r: np.ndarray,
        angle: np.ndarray,
    ):
        u = self.v_y * np.cos(angle) - self.v_x * np.sin(angle)   # scalar product with tangential versor (-sin(a), cos(a))
        du_x = derivative(u, 0, self.Dx)     # compute gradient of u
        du_y = derivative(u, 1, self.Dx)
        du_r = du_x * np.cos(angle) + du_y * np.sin(angle)  # scalar product with radial versor (cos(a), sin(a))
        return u/r + du_r

    def plot_field(
        self,
        time = None,
        savepath = None,
        col_map = None,
        streamplt_col='purple',
    ):
        if col_map is None:
            col_map = cm.BrBG_r
            lim = np.max(np.abs(self.q))
            qmax = lim
            qmin = -lim
        else:
            qmax = np.max(self.q)
            qmin = np.min(self.q)
        # lin_thresh = np.power(10, np.floor(np.log10(lim/100)))  # power of 10 closest to lim/100
        # log_norm = SymLogNorm(linthresh=lin_thresh, vmin=-lim, vmax=lim)
        # col_map = cm.PuOr_r
        
        fig, ax = plt.subplots()
        (x, y) = coordinates(self.Dx, self.N)
        # contour_plt = ax.contourf(x.T, y.T, self.q.T, norm=log_norm, cmap=col_map, levels=75)
        levs = np.linspace(qmin, qmax, 75)
        contour_plt = ax.contourf(x.T, y.T, self.q.T, cmap=col_map, levels=levs)
        cbar = fig.colorbar(contour_plt, ax=ax)
        
        ax.streamplot(x.T, y.T, *(v.T for v in self.velocity()), color=streamplt_col)
        
        theta = np.linspace(0, (self.N-1) * self.Dx, num=5)
        labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
        ax.set_xticks(theta, labels)
        ax.set_yticks(theta, labels)
        ax.set_xlim(0, (self.N-1) * self.Dx)
        ax.set_ylim(0, (self.N-1) * self.Dx)
        ax.set_aspect('equal')
        if time is not None:
            ax.set_title(f'Potential Vorticity  $|$  t={time:.3f}')
        if savepath is not None:
            fig.savefig(savepath, dpi=200)
        else:
            plt.show()
        plt.close(fig)

    def arakawa_jacobian(
        self,
    ):
        if self.upd_J:
            # the neighbours of the i-th point are named by numbers according to the following order:
            #    y
            #    ^ 8 1 2
            #    | 7 i 3
            #    | 6 5 4
            #   -|------> x
            #
            q  = self.q
            q1 = np.roll(q,  -1, axis=1)
            q2 = np.roll(q1, -1, axis=0)
            q3 = np.roll(q,  -1, axis=0)
            q4 = np.roll(q3, +1, axis=1)
            q5 = np.roll(q,  +1, axis=1)
            q6 = np.roll(q5, +1, axis=0)
            q7 = np.roll(q,  +1, axis=0)
            q8 = np.roll(q7, -1, axis=1)
            p  = self.streamfunction()
            p1 = np.roll(p,  -1, axis=1)
            p2 = np.roll(p1, -1, axis=0)
            p3 = np.roll(p,  -1, axis=0)
            p4 = np.roll(p3, +1, axis=1)
            p5 = np.roll(p,  +1, axis=1)
            p6 = np.roll(p5, +1, axis=0)
            p7 = np.roll(p,  +1, axis=0)
            p8 = np.roll(p7, -1, axis=1)
            self.J = -((p5 + p4 - p1 - p2) * (q3 - q)
                      +(p6 + p5 - p8 - p1) * (q - q7)
                      +(p3 + p2 - p7 - p8) * (q1 - q)
                      +(p4 + p3 - p6 - p7) * (q - q5)
                      +(p3 - p1) * (q2 - q)
                      +(p5 - p7) * (q - q6)
                      +(p1 - p7) * (q8 - q)
                      +(p3 - p5) * (q - q4)) / (12 * self.Dx**2)
            self.upd_J = False
        return self.J
    
    def qsum20(self):
        self.q -= self.q.sum()
        self.upd_psi = True
        self.upd_J = True
        self.upd_v = True

    def __neg__(self):
        neg_self = self.__class__(vorticity = -self.q)
        if self.psi is not None:
            neg_self.psi = -self.psi
        if self.v_x is not None:
            (neg_self.v_x, neg_self.v_y) = (-self.v_x, -self.v_y)
        return neg_self
    
    def __add__(self, other):
        sum_ = self.__class__(vorticity = self.q + other.q)
        if (self.psi is not None) and (other.psi is not None):
            sum_.psi = self.psi + other.psi
        if (self.v_x is not None) and (other.v_x is not None):
            (sum_.v_x, sum_.v_y) = (self.v_x + other.v_x, self.v_y + other.v_y)
        return sum_

    def __sub__(self, other):
        diff = self.__class__(vorticity = self.q - other.q)
        if (self.psi is not None) and (other.psi is not None):
            diff.psi = self.psi - other.psi
        if (self.v_x is not None) and (other.v_x is not None):
            (diff.v_x, diff.v_y) = (self.v_x - other.v_x, self.v_y - other.v_y)
        return diff

    def __repr__(self):
        psi_exist = (self.psi is not None)
        J_exist = (self.J is not None)
        v_exist = (self.v_x is not None)
        return f'<{self.__class__.__name__}> instance of size {self.q.shape} with attributes [q{", psi" if psi_exist else ""}{", J" if J_exist else ""}{", v_x, v_y" if v_exist else ""}].'


class FluidStateTopography(FluidState):
    def __init__(
        self,
        topography: np.ndarray,
        simul = None,
        pot_vorticity = None,
        rel_vorticity = None,
        f_Coriolis = None,
    ):
        if simul is None:
            self.init_topography(topography)
        else:
            if simul.reload_bak:
                self.topo = np.load(simul.bak_dir / 'topography.npy')
            else:
                self.init_topography(topography)
                if simul.diagnostics:
                    np.save(simul.bak_dir / 'topography.npy', self.topo)
        # self.h = 1 - 2/3 * self.topo/self.topo.max()
        self.h = 1 - self.topo/np.abs(self.topo).max() / self.hb_scl
        # gradient terms computed along diagonals, divided by h (non-linear term in q[psi])
        self.Hmd = (np.roll(np.roll(self.h, -1, axis=0), -1, axis=1) - np.roll(np.roll(self.h, +1, axis=0), +1, axis=1)) / (8*self.Dx**2 * self.h)
        self.Hsd = (np.roll(np.roll(self.h, +1, axis=0), -1, axis=1) - np.roll(np.roll(self.h, -1, axis=0), +1, axis=1)) / (8*self.Dx**2 * self.h)

        super().__init__(simul, pot_vorticity, rel_vorticity, f_Coriolis)

    def init_topography(
        self,
        topography,
    ):
        if topography.shape == (self.N, self.N):
            self.topo = topography
        else:
            raise Exception("Error: wrong argument: topography must be <numpy.ndarray> of shape (N, N).")

    def init_vorticity(
        self,
        pot_vorticity,
        rel_vorticity,
    ):
        if rel_vorticity is not None:
            q = (rel_vorticity + self.f_Coriolis) / self.h
            if (pot_vorticity is not None):
                raise Exception("Error: wrong arguments: cannot set both potential and relative vorticity.")
        else:
            q = pot_vorticity

        if q is not None:
            if isinstance(q, np.ndarray) and q.shape == (self.N, self.N):
                self.q = q
            else:
                raise Exception("Error: wrong argument: q must be <numpy.ndarray> of shape (N, N).")
        else:
            self.q = self.f_Coriolis / self.h

    def streamfunction(
        self,
    ):
        if self.upd_psi:
            n_iter = 0
            tol = 1e-15
            err = 1.
            zh = (self.q*self.h - self.f_Coriolis) * self.h
            if self.psi is None:
                self.psi = -inv_laplacian2d(zh, self.Dx)
            psi_p = np.empty_like(self.psi)
            while err > tol:
                n_iter += 1
                np.copyto(psi_p, self.psi) # memory-safe version of `psi_p = self.psi`
                Hpsi = self.Hmd * (np.roll(np.roll(self.psi, -1, axis=0), -1, axis=1) - np.roll(np.roll(self.psi, +1, axis=0), +1, axis=1)) + self.Hsd * (np.roll(np.roll(self.psi, +1, axis=0), -1, axis=1) - np.roll(np.roll(self.psi, -1, axis=0), +1, axis=1))
                np.copyto(self.psi, inv_laplacian2d(Hpsi - zh, self.Dx))
                err = np.abs(self.psi - psi_p).max() / (np.abs(self.psi).max() + 1e-9)
            if n_iter > 30: print(f'{n_iter} iterations were required.')
            self.upd_psi = False
        return self.psi

    def velocity(
        self,
    ):
        if self.upd_v:
            self.streamfunction()
            self.v_x = derivative(self.psi, 1, self.Dx) / self.h  # d(psi)/dy / h
            self.v_y = -derivative(self.psi, 0, self.Dx) / self.h # -d(psi)/dx / h
            self.upd_v = False
        return self.v_x, self.v_y

    def vorticity(self):
        return self.q*self.h - self.f_Coriolis

    def dissipation(self):
        return pseudo_laplacian2d(self.vorticity()) / self.h

    def energy(self):
        return integrate(self.vorticity() * self.streamfunction() * self.h, self.Dx) / 2
        # self.velocity()
        # return integrate((self.v_x**2 + self.v_y**2) * self.h, self.Dx) / 2

    def casimir(
        self,
        pow: int,
    ):
        return integrate(np.power(self.q, pow) * self.h, self.Dx) / pow
    
    def plot_field(
        self,
        time = None,
        savepath = None,
        col_map = None,
        streamplt_col='purple',
    ):
        # col_map = cm.PuOr_r
        
        fig, ax = plt.subplots()
        (x, y) = coordinates(self.Dx, self.N)
        ax.contour(x.T, y.T, self.topo.T, colors='black', alpha=0.75)
        if col_map is None:
            col_map = cm.BrBG_r
            lim = np.max(np.abs(self.q))
            qmax = lim
            qmin = -lim
        else:
            qmax = np.max(self.q)
            qmin = np.min(self.q)
        # lin_thresh = np.power(10.,np.floor(np.log10(lim/100)))  # closest power of 10
        # log_norm = SymLogNorm(linthresh=lin_thresh, vmin=-lim, vmax=lim)
        levs = np.linspace(qmin, qmax, 75)
        contour_plt = ax.contourf(x.T, y.T, self.q.T, cmap=col_map, levels=levs)
        # contour_plt = ax.contourf(x.T, y.T, self.q.T, norm=log_norm, cmap=col_map, levels=75)
        cbar = fig.colorbar(contour_plt, ax=ax)
        
        ax.streamplot(x.T, y.T, *(v.T for v in self.velocity()), color=streamplt_col)
        
        theta = np.linspace(0, (self.N-1) * self.Dx, num=5)
        labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
        ax.set_xticks(theta, labels)
        ax.set_yticks(theta, labels)
        ax.set_xlim(0, (self.N-1) * self.Dx)
        ax.set_ylim(0, (self.N-1) * self.Dx)
        ax.set_aspect('equal')
        if time is not None:
            ax.set_title(f'Potential Vorticity  $|$  t={time:.3f}')
        if savepath is not None:
            fig.savefig(savepath, dpi=200)
        else:
            plt.show()
        plt.close(fig)
        
    def arakawa_jacobian(
        self,
    ):
        if self.upd_J:
            # 1. Call base arakawa_jacobian, which:
            #    - computes the base value
            #    - sets upd_J to False
            super().arakawa_jacobian()
            # 2. Overwrite the base Jacobian with the topographic one
            self.J /= self.h
        return self.J
    
    def qsum20(self):
        self.q -= self.vorticity().sum() / self.h.sum()
        self.upd_psi = True
        self.upd_J = True
        self.upd_v = True

    def __neg__(self):
        neg_self = self.__class__(vorticity = -self.q, topography=self.topo)
        if self.psi is not None:
            neg_self.psi = -self.psi
        if self.v_x is not None:
            (neg_self.v_x, neg_self.v_y) = (-self.v_x, -self.v_y)
        return neg_self
    
    def __add__(self, other):
        sum_ = self.__class__(vorticity = self.q + other.q, topography=self.topo)
        if (self.psi is not None) and (other.psi is not None):
            sum_.psi = self.psi + other.psi
        if (self.v_x is not None) and (other.v_x is not None):
            (sum_.v_x, sum_.v_y) = (self.v_x + other.v_x, self.v_y + other.v_y)
        return sum_

    def __sub__(self, other):
        diff = self.__class__(vorticity = self.q - other.q, topography=self.topo)
        if (self.psi is not None) and (other.psi is not None):
            diff.psi = self.psi - other.psi
        if (self.v_x is not None) and (other.v_x is not None):
            (diff.v_x, diff.v_y) = (self.v_x - other.v_x, self.v_y - other.v_y)
        return diff


class VorticityPlotter:
    """
    Reuse a single Figure/Axes/Colorbar for the whole run, rendered off-screen
    with Agg (so no GUI windows pop up). Does not affect global Matplotlib backend.
    """
    def __init__(
        self,
        Dx: float,  # grid spacing
        n: int,     # grid size
        cmap = cm.BrBG_r):
        self.cmap = cmap

        self.fig, self.ax = plt.subplots()
        FigureCanvasAgg(self.fig) # Attach an Agg canvas (off-screen renderer)

        theta = np.linspace(0, (n - 1) * Dx, num=5)
        labels = [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$']
        self.ax.set_xticks(theta, labels)
        self.ax.set_yticks(theta, labels)
        self.ax.set_xlim(0, (n - 1) * Dx)
        self.ax.set_ylim(0, (n - 1) * Dx)
        self.ax.set_aspect('equal')

        self.im = self.ax.imshow( # Initial image with dummy data
            np.zeros((n, n), dtype=float).T,
            origin='lower',
            extent=(0, (n-1)*Dx, 0, (n-1)*Dx),
            cmap=self.cmap,
            # norm=SymLogNorm(linthresh=1e-3, vmin=-1.0, vmax=1.0),
            vmin=-1.0, vmax=1.0,
            interpolation='nearest',
        )
        self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        self.title_obj = self.ax.set_title("Potential Vorticity | t=0.000")

    def update(self,
            q,
            time: float,
            savepath = None,
            upd_clim = True
        ):
        """Update image data in place and save off-screen frame."""
        self.im.set_data(q.T)

        if upd_clim and (np.count_nonzero(q) > 0):
            lim = float(np.max(np.abs(q)))
            p10lim = np.power(10, np.floor(np.log10(lim)))
            lim = np.ceil(lim/p10lim)*p10lim
            # lin_thresh = np.power(10.,np.floor(np.log10(lim/10)))
            # self.im.set_norm(SymLogNorm(linthresh=lin_thresh, vmin=-lim, vmax=lim))
            self.im.set_norm(Normalize(vmin=-lim, vmax=lim))
            self.cbar.update_normal(self.im)

        self.title_obj.set_text(f"Potential Vorticity | t={time:.3f}")

        if savepath is not None:
            self.fig.savefig(savepath, dpi=200)

    def close(self):
        try:
            self.cbar.remove()
        except Exception:
            pass
        self.fig.clf()
        plt.close(self.fig)


def energy_spectrum(
    vx: np.ndarray,
    vy: np.ndarray,
):
    n = vx.shape[0]
    mod_k = modulus_k(n)
    k_max = np.ceil(np.max(mod_k))
    vx_fft = ft.rfft2(vx, norm='ortho') / n
    vy_fft = ft.rfft2(vy, norm='ortho') / n
    E_k = (np.abs(vx_fft)**2 + np.abs(vy_fft)**2) / 2
    E_k[:, 1:-1] *= 2
    spectrum, _, _ = binned_statistic(mod_k.flatten(), E_k.flatten(), statistic='mean', bins=np.arange(k_max+1))
    return (2*np.pi)**3 * np.arange(k_max) * spectrum


@cache
def zero_forcing(
    n: int,     # grid size
    *args,**kwargs,
):
    return np.zeros((n, n))


def random_scalefree_field(
    n: int,     # grid size
    energy = 1.,
    *args,**kwargs,
):
    power_spectrum = np.sqrt(energy) * scalefree_spectrum(n)
    random_phase = np.random.rand(n, n//2+1)
    F_ft = power_spectrum * np.exp(2j*np.pi * random_phase)
    return ft.irfft2(F_ft, norm='ortho', s=(n,n)) * n/(2*np.pi)


@cache
def scalefree_spectrum(
    n: int,     # grid size
):
    k = modulus_k(n)
    with np.errstate(divide='ignore'):
        f2 = np.power(k, -2) / (2*np.pi * k)
    f2[k==0.] = 0.
    norm = f2.sum()
    f2[:, 1:-1] /= 2
    return np.sqrt(f2 / norm)


def random_forcing(
    n: int,     # grid size
    Dt,
    trg_eps = 1.,
):
    k_F = 50. if n > 256 else 25.
    power_spectrum = vorticity_forcing_spectrum(trg_eps, k_F, n)
    random_phase = np.random.rand(n, n//2+1)
    F_ft = power_spectrum * np.exp(2j*np.pi * random_phase)
    return ft.irfft2(F_ft, norm='ortho', s=(n,n)) * n/(2*np.pi) / np.sqrt(Dt)


@cache
def vorticity_forcing_spectrum(
    strength: float,
    k_F: float,
    n: int,     # grid size
):
    band_size = 3.
    band_max = k_F + band_size/2.
    band_min = k_F - band_size/2.
    k = modulus_k(n)
    f2 = 2*strength * np.logical_and(band_min < k, k < band_max) / (2*np.pi * band_size)
    f2[:, 1:-1] /= 2
    return np.sqrt(k * f2)


@cache
def modulus_k(
    n: int,     # grid size
):
    kx = np.arange(n)
    kx[n//2+1:n] = kx[n-n//2-1:0:-1]
    ky = np.arange(n//2+1)
    return np.sqrt(kx[:,None]**2+ky[None,:]**2)


def gauss_topography(
    Dx: float,  # grid spacing
    n: int,     # grid size
    sigma = None,  # mount width
):
    ctr = (n-1)//2 * Dx
    if sigma is None:
        sigma = 2*np.pi / 10
    dist, _ = relative_pos(ctr, ctr, Dx, n)
    return np.exp(-dist**2 / (2 * sigma**2))


def diag_ridge_topography(
    Dx: float,  # grid spacing
    n: int,     # grid size
    sigma = None,  # mount width
):
    sd_len = n * Dx
    pos = np.arange(n) * Dx
    xy_diff = np.abs(pos[:, None] - pos[None, :])
    dist2diag = np.where(xy_diff < sd_len/2, xy_diff, sd_len - xy_diff) / np.sqrt(2)
    if sigma is None:
        sigma = 2*np.pi / 10
    return np.exp(-dist2diag**2 / (2 * sigma**2))


def random_topography(
    peaks: int, # number of peaks
    Dx: float,  # grid spacing
    n: int,     # grid size
):
    topo = np.zeros((n, n))
    for i in range(peaks):
        sign = [-1, 1][np.random.randint(2)]
        height = np.exp(np.random.rand()-1)
        fatness = 1.3* height * (1.5 * np.random.rand() + 0.25)
        ctr = 2*np.pi * np.random.rand(2)
        dist, _ = relative_pos(*ctr, Dx, n)
        topo += sign * height * np.exp(-(dist/fatness)**2)
    topo /= topo.max()
    return topo


def square_wells_topography(
    Dx: float,  # grid spacing
    n: int,     # grid size
    sigma = None,  # mount width
):
    ctr = np.array(((np.pi/2, np.pi/2), (np.pi/2, 3*np.pi/2), (3*np.pi/2, np.pi/2), (3*np.pi/2, 3*np.pi/2)))
    if sigma is None:
        sigma = 2*np.pi / 5
    sign = (+1, -1, -1, +1)
    topo = np.zeros((n, n))
    for i in range(4):
        dist, _ = relative_pos(*ctr[i], Dx, n)
        topo += sign[i] * np.exp(-dist**2 / (2*sigma**2))
    max_heigth = np.max(topo)
    return topo/max_heigth


def eig_function(
    h: np.ndarray,  # topography
    n: int,         # grid size
):
    from scipy.sparse.linalg import eigs
    q_of_psi = (laplacian_matrix(n) / h.flatten()).T
    e, ev = eigs(q_of_psi, 64, which='SM', tol=0, maxiter=1e14)
    del(q_of_psi)
    return e, ev


def laplacian_matrix(
    n: int, # grid size
):
    laplacian_1d = 2 * np.eye(n) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1) - np.diag(np.ones(1), n-1) - np.diag(np.ones(1), 1-n)
    laplacian_2d = np.kron(laplacian_1d, np.eye(n)) + np.kron(np.eye(n), laplacian_1d) 
    return laplacian_2d


def laplacian_spmat(
    Dx: float,  # grid spacing
    n: int,     # grid size
):
    import scipy.sparse as sp
    laplacian1d = sp.diags([-2] + [1]*4, offsets=[0, 1, -1, n-1, 1-n], shape=(n, n)) / Dx**2
    laplacian2d = sp.kron(laplacian1d, sp.eye(n)) + sp.kron(sp.eye(n), laplacian1d)
    return laplacian2d


def dx_spmat(
    Dx: float,  # grid spacing
    n: int,     # grid size
):
    import scipy.sparse as sp
    dx1d = sp.diags([1]*2 + [-1]*2, offsets=[1, 1-n, -1, n-1], shape=(n, n)) / (2*Dx)
    dx2d = sp.kron(dx1d, sp.eye(n))
    return dx2d


def dy_spmat(
    Dx: float,  # grid spacing
    n: int,     # grid size
):
    import scipy.sparse as sp
    dy1d = sp.diags([1]*2 + [-1]*2, offsets=[1, 1-n, -1, n-1], shape=(n, n)) / (2*Dx)
    dy2d = sp.kron(sp.eye(n), dy1d)
    return dy2d


def relative_pos(
    x_ctr: float,   # center coordinates
    y_ctr: float,
    Dx: float,      # grid spacing
    n: int,         # grid size
):
    sd_len = n * Dx
    pos = np.arange(n) * Dx
    ctr = sd_len / 2
    x_dist = (pos - x_ctr + ctr) % sd_len - ctr
    y_dist = (pos - y_ctr + ctr) % sd_len - ctr
    dist2ctr = np.sqrt(np.square(x_dist)[:, None] + np.square(y_dist)[None, :])
    angle = np.arctan2(y_dist[None, :], x_dist[:, None])
    return dist2ctr, angle


def inv_laplacian2d(
    omega: np.ndarray,
    Dx: float,  # grid spacing
):
    omega_ft = ft.rfft2(omega)
    psi_ft = omega_ft / laplacian_eig(omega_ft.shape, Dx)
    return ft.irfft2(psi_ft, s=omega.shape)


@cache
def laplacian_eig(
    shape: tuple,
    Dx: float,  # grid spacing
):
    cos_sum = np.cos(np.arange(shape[0]) * Dx)[:, None] + np.cos(np.arange(shape[1]) * Dx)[None, :]
    lap_eig = 2* (cos_sum - 2) / Dx**2
    lap_eig[0, 0] = np.inf
    return lap_eig


def derivative(
    f: np.ndarray,
    axis: int,
    Dx: float,  # grid spacing
) -> np.ndarray:
    return (np.roll(f, -1, axis=axis) - np.roll(f, +1, axis=axis)) / (2 * Dx)


def pseudo_laplacian2d(
    f: np.ndarray,
):
    return np.roll(f, -1, axis=0) + np.roll(f, +1, axis=0) + np.roll(f, +1, axis=1) + np.roll(f, -1, axis=1) - 4* f


def autocorrelation(
    f: np.ndarray,  # field
):
    f_T = ft.rfft2(f)
    return ft.irfft2(f_T * f_T.conjugate(), s=f.shape)
    

def pbc_dist(
    x: int, # position
    D: int, # domain size
):
    return min(x, D-x)


def measure_valley_dist(
    topo: np.ndarray,  # topography
    n: int,         # domain size
):
    C_self = autocorrelation(topo)
    max_dist = int(np.round(n//np.sqrt(2)))
    C = np.zeros(max_dist + 1)
    count = np.zeros(max_dist + 1)

    for x in range(n):
        for y in range(n):
            dist = np.hypot(pbc_dist(x, n), pbc_dist(y, n)).round().astype(int)
            count[dist] += 1
            C[dist] += C_self[x, y]
    C /= count * np.square(topo).sum()
    dist = np.arange(max_dist + 1)
    return C.sum(where=(dist > n/2)) / (max_dist - n/2)


def measure_concentration(
    f: np.ndarray,  # field
    n: int,         # domain size
):
    f_flt = gaussian_filter(f, sigma=n/50, truncate=2., mode='wrap')
    L2_measure = np.square(f_flt).sum()
    L1_measure = np.abs(f_flt).sum()
    concentration =  L2_measure * n**2 / L1_measure**2
    return 1 - 1 / concentration


def zero_coord( # finding combined zeros in the square domain [a, b]^2
    f_x: np.ndarray,
    f_y: np.ndarray,
    a: int, # interval left endpoint
    b: int, # interval right endpoint
):
    coor = np.arange(a, b-1)
    (X, Y) = np.meshgrid(coor, coor, indexing='ij')
    is_zero_fx_y = f_x[a:b, a:b-1] * f_x[a:b, a+1:b] <= 0
    is_zero_fy_x = f_y[a:b-1, a:b] * f_y[a+1:b, a:b] <= 0
    is_zero_f = np.logical_and(np.logical_or(is_zero_fx_y[0:-1, :], is_zero_fx_y[1:, :]), np.logical_or(is_zero_fy_x[:, 0:-1], is_zero_fy_x[:, 1:]))
    (zero_x, zero_y) = (0.,0.)
    count = 0
    for x, y in zip(X[is_zero_f], Y[is_zero_f]):
        (zero_x, zero_y) = (zero_x + x+0.5, zero_y + y+0.5)
        count = count + 1
    if count != 0:
        return zero_x/count, zero_y/count
    else:
        return (a+b)/2, (a+b)/2


def avg_coord( # averaging in the square domain [a, b]^2
    f: np.ndarray,      # weights
    a: int, # interval left endpoint
    b: int, # interval right endpoint
):
    (x, y) = coordinates()
    avg_x = np.average(x[a:b, a:b], weights=f[a:b, a:b])
    avg_y = np.average(y[a:b, a:b], weights=f[a:b, a:b])
    return avg_x, avg_y


def neighboring_clusters(
    x: int,         # point coordinates
    y: int,
    f: np.ndarray,  # field
    n: int,         # domain size
):
    neighbors = []
    if x != 0:
        c = f[x-1, y]
        if c >= 0:
            neighbors.append(c)
    if y != 0:
        c = f[x, y-1]
        if c >= 0 and c not in neighbors:
            neighbors.append(c)
    if x == n-1:
        c = f[0, y]
        if c >= 0 and c not in neighbors:
            neighbors.append(c)
    if y == n-1:
        c = f[x, 0]
        if c >= 0 and c not in neighbors:
            neighbors.append(c)
    return sorted(neighbors)


def cluster_ones(
    f: np.ndarray,  # field
    n: int,         # domain size
):
    cluster = []
    f_c = -np.ones((n, n)).astype(int)
    for x in range(n):
        for y in range(n):
            if f[x, y] == 1:
                neighbors = neighboring_clusters(x, y, f_c, n)
                if len(neighbors) == 0:
                    f_c[x, y] = len(cluster)
                    cluster.append([[x,y]])
                else:
                    f_c[x, y] = neighbors[0]
                    cluster[neighbors[0]].append([x,y])
                if len(neighbors) > 1:
                    for i in range(1, len(neighbors)):
                        for node in cluster[neighbors[i]]:
                            f_c[tuple(node)] = neighbors[0]
                        cluster[neighbors[0]].extend(cluster[neighbors[i]])
                        del cluster[neighbors[i]]
                        f_c[f_c > neighbors[i]] -= 1
                        for j in range(i+1, len(neighbors)):
                            neighbors[j] -= 1
    return sorted(cluster, key=len, reverse=True)


def largest_valley(
    topo: np.ndarray,
    n: int,     # domain size
    thresh = 0.2,
):
    valley_thresh = thresh * topo.max() + (1 - thresh) * topo.min()
    valleys = np.where(topo < valley_thresh, np.ones((n, n)), np.zeros((n, n)))
    cluster = cluster_ones(valleys, n)
    rel_size = len(cluster[0]) / n**2
    x_avg = y_avg = count = 0
    for point in cluster[0]:
        x_avg += point[0]
        y_avg += point[1]
        count += 1
    x_avg /= count
    y_avg /= count
    # print(f'The largest valley is located at ({x_avg:.1f}, {y_avg:.1f}) and fills about {rel_size*100:.1f}% of the domain.')
    return rel_size, (x_avg, y_avg)


def optimal_rescale(f, g):
    from scipy.optimize import minimize
    def Q(alpha, f, g):
        return np.square(g - alpha*f).sum()
    def Q_Jac(alpha, f, g):
        return (f * (alpha*f - g)).sum()
    def Q_Hess(alpha, f, g):
        return np.square(f).sum()
    alpha0 = (g.max() / f.max() + g.min() / f.min()) / 2
    result = minimize(fun=Q, jac=Q_Jac, hess=Q_Hess, x0=alpha0, args=(f, g), method='Newton-CG')
    if result['success'] is True:
        return result['x']
    else:
        return None


def state_diff(f, g):
    if (f * g).sum() < 0:
        g = -g
    alpha = optimal_rescale(f, g)
    if alpha is not None:
        print(f'The relative difference between states is {np.abs(alpha*f - g).sum() / np.abs(g).sum()}.')
        max_diff = np.abs(alpha*f - g).max()
        plt.contourf((alpha*f.T - g.T)/max_diff, levels=np.linspace(-1,1,50))
        plt.colorbar()
        plt.show()
    else:
        print('Error: could not compare states.')


def lamb_dipole(
    Dx: float,  # grid spacing
    n: int,     # grid size
    orient = np.pi/2,
    U = 2.,
):
    radius = 0.73
    c_0 = 3.83170597020751231
    K = c_0 / radius
    C_L = - 2. * K * U / sp.j0(c_0)
    ctr = n//2 - 0.5
    dist2axis = np.arange(n) - ctr
    dist2ctr = np.sqrt(np.square(dist2axis)[:, None] + np.square(dist2axis)[None, :]) * Dx
    angle = np.arctan2(dist2axis[None, :], dist2axis[:, None])
    dipole = C_L * sp.j1(K*dist2ctr) * np.sin(angle - orient)
    dipole[dist2ctr > radius] = 0
    return dipole


def find_q(
    directory,
    idx=-1,
):
    q_files = sorted(directory.glob('q*.npz'),
                        key = lambda path: int(path.stem[1:])
                    )
    if not q_files:
        return None
    return q_files[idx]


def open_folder(
    folder,
    overwrite = False,
):
    if folder.exists():
        if overwrite:
            for file in folder.iterdir():
                if not file.is_dir():
                    file.unlink()
    else:
        folder.mkdir()


@cache
def coordinates(
    Dx: float,  # grid spacing
    n: int,     # grid size
):
    x = np.arange(0, n) * Dx
    y = np.arange(0, n) * Dx
    return np.meshgrid(x, y, indexing='ij')

def integrate(
    f:np.ndarray,   # integrand
    Dx: float,      # grid spacing
):
    return np.sum(f) * Dx**2