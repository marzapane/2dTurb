from numpy import pi, inf

T = int(1e+6)
N = 512
Re = 1e+5
ekman = 5e-5
eps = 1e+0
# cfg_name = 'steadE100'
cfg_name = 'turb+0'
sd_len = 2*pi # domain size
Dx = sd_len / N
Dt = 1 * Dx
f_Coriolis = 0.
hb_scale = 10.
