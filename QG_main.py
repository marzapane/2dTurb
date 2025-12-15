#!/opt/intel/oneapi/intelpython/latest/bin/python
# #!/usr/bin/env python

import argparse
import numpy as np
import po2d_lib as po
    
def randIC(Energy, Dx, N, topo=None):
    q = po.random_topography(40, Dx, N)
    # q = (q - q[:, N-1::-1]) / 2
    # q = (q - q[N-1::-1, :]) / 2
    if topo is None:
        fluid = po.FluidState(rel_vorticity=q)
    else:
        fluid = po.FluidStateTopography(topo, rel_vorticity=q)
    fluid.qsum20()
    fluid.streamfunction(0)
    E = fluid.energy()
    q = fluid.q * np.sqrt(Energy / E)
    del fluid
    return q

def main(args):
    simul = po.FluidSimulator(po.random_forcing, adaptive_Dt=True)
    # simul = po.FluidSimulator(po.zero_forcing, adaptive_Dt=True)
    simul.silent = args.silent
    simul.initialize_run_state()
    topo = po.gauss_topography(simul.Dx, simul.N)
    # energy = 100.
    # fluid = po.FluidStateTopography(topo, simul, rel_vorticity=randIC(energy, simul.Dx, simul.N, topo))
    fluid = po.FluidStateTopography(topo, simul)
        
    simul.set_physical_param(fluid)
    for t in simul.time_exec:
        simul.advance_dt(fluid, t)

        # # Fix energy
        # E = fluid.energy()
        # if (E - energy) > 1e-5:
        #     print(f'At time {simul.time:g} kynetic energy increased to E={E:g}.')
        # fluid.q *= np.sqrt(energy/E)

    simul.conclude()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with or without progress bar.")
    parser.add_argument('--silent', action='store_true', help="Run in silent mode without progress bar, printing info to stdout.")
    main(parser.parse_args())
