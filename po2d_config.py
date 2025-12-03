from numpy import pi, inf

T = int(1e+6)
N = 128
Re = 1e+3
ekman = 0#5e-5
eps = 1e+4
# cfg_name = 'steadE100'
cfg_name = 'rot001'
sd_len = 2*pi # domain size
Dx = sd_len / N
Dt = 1 * Dx
