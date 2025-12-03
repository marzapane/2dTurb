#!/usr/bin/env python

import numpy as np
import pandas as pd

def vertical_flip(square):
    n = square.shape[0]
    return square[:, n-1::-1]

def horizontal_flip(square):
    n = square.shape[0]
    return square[n-1::-1, :]

def main_diagonal_flip(square):
    n = square.shape[0]
    return square[n-1::-1, n-1::-1].transpose()    

def secondary_diagonal_flip(square):
    return square.transpose()

def clockwise_rotation(square):
    n = square.shape[0]
    return square.transpose()[:, n-1::-1]

def counter_clockwise_rotation(square):
    n = square.shape[0]
    return square.transpose()[n-1::-1, :]

def pi_rotation(square):
    n = square.shape[0]
    return square[n-1::-1, n-1::-1]

# def main():
#     data = np.zeros((64, 6))
#     final_states = np.load('../Results/decay_table.npz')['fin']
#     for i in range(0, 64):
#         eig_fn = np.load('/home/bilottos/Documents/PhD/Thesis project/Programs/2DPO/Results/Gauss128Eig/EigFn.npy')[i]
#         data[i, 0] = i
#         data[i, 1] = final_states[i]
#         data[i, 2] = (eig_fn * vertical_flip(eig_fn)).sum()
#         data[i, 3] = (eig_fn * horizontal_flip(eig_fn)).sum()
#         data[i, 4] = (eig_fn * main_diagonal_flip(eig_fn)).sum()
#         data[i, 5] = (eig_fn * secondary_diagonal_flip(eig_fn)).sum()
#     dataframe = pd.DataFrame(data, columns=['eigenstate', 'decays in', 'm-', 'm|', 'm/', 'm\\'])
#     dataframe = dataframe.round(2)
#     dataframe = dataframe.astype({'eigenstate':'int', 'decays in':'int'})
#     print(dataframe.to_latex(index=False))

def tex_table():
    data = np.zeros((64, 4))
    eig_system = np.load('../results/eigfn_sym.npz')['eigfn']
    final_states = np.load('../results/decay_table.npz')['fin']
    for i in range(64):
        eig_fn = eig_system[i]
        data[i, 0] = i
        data[i, 1] = (eig_fn * counter_clockwise_rotation(eig_fn)).sum()
        data[i, 2] = (eig_fn * secondary_diagonal_flip(eig_fn)).sum()
        # data[i, 2] = (eig_fn * horizontal_flip(eig_fn)).sum()
        data[i, 3] = final_states[i]
    dataframe = pd.DataFrame(data, columns=['eigenstate', 'decays in', 'm^r', 'm^\\'])
    dataframe = dataframe.round(2)
    dataframe = dataframe.astype({'eigenstate':'int', 'decays in':'int'})
    print(dataframe.to_latex(index=False))
    
def main():
    eig_system = np.load('../results/hill_eigfn.npz')['eigfn']
    print(' i   <q>   m^90  m^/  m^\\  m^|  m^-   #')
    for i in range(64):
        eig_fn = eig_system[i+1]
        (Nx, Ny) = eig_fn.shape
        norm = np.square(eig_fn).sum()
        print(f'{i+1:02}', end='  ')
        print(f'{eig_fn.sum() / (Nx*Ny):5.2f}', end='  ')
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

if __name__ == "__main__":
    main()