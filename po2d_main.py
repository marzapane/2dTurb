#!/usr/bin/env python

import argparse
import numpy as np
import po2d_lib as po

def main(args):
    analize_vortex = False
    simul = po.FluidSimulator(po.zero_forcing, analize_vortex, adaptive_Dt=True)
    simul.silent = args.silent
    simul.initialize_run_state()
    fluid = po.FluidState(simul.reload_bak, simul.bak_dir, simul.bak_file, vorticity=np.load('../results/hill_eigfn.npz')['eigfn'][61])
    simul.set_physical_param(fluid)
    for t in simul.time_exec:
        simul.advance_dt(fluid, t)
    simul.conclude()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation with or without progress bar.")
    parser.add_argument('--silent', action='store_true', help="Run in silent mode without progress bar, printing info to stdout.")
    main(parser.parse_args())