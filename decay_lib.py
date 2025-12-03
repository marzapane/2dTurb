def plot_vorticity_file(
    file,
    title = None
):
    data = np.load(file)
    plt.contourf(data['q'])
    plt.colorbar()
    if title is not None:
        plt.title(title)
    plt.show()
    plt.close()

def last_file(dir):
    max_number = -1
    max_file = None
    for file in dir.glob('q*.npz'):
        try:
            file_number = int(file.stem[1:])
        except ValueError:
            print(f'<{file.name}.npz> does not have a valid name.')
            pass
        else:
            if file_number > max_number:
                max_number = file_number
                max_file = file
    return max_file, max_number

def first_file(dir):
    min_number = np.inf
    min_file = None
    for file in dir.glob('q*.npz'):
        try:
            file_number = int(file.stem[1:])
        except ValueError:
            print(f'<{file.name}.npz> does not have a valid name.')
            pass
        else:
            if file_number < min_number:
                min_number = file_number
                min_file = file
    return min_file, min_number

def extract_file_list(dir):
    files = [(int(path.stem[1:]), path) for path in dir.glob('q*.npz')]
    files.sort(key=lambda x: x[0])
    return files

def time_diff(dir):
    files = extract_file_list(dir)
    time = [np.load(path)['t'] for num, path in files]
    return np.array(time)

def avg_diff(dir):
    file0, _ = first_file(dir)
    files = extract_file_list(dir)
    diff = []
    q_p = np.load(file0)['q']
    for num, path in files:
        q = np.load(path)['q']
        diff.append((q-q_p).mean())
        q_p = q
    return np.array(diff)

def std_diff(dir):
    file0, _ = first_file(dir)
    files = extract_file_list(dir)
    diff_std = []
    q_p = np.load(file0)['q']
    for num, path in files:
        q = np.load(path)['q']
        diff_std.append((q-q_p).std())
        q_p = q
    return np.array(diff_std)

def std_decay(dir):
    file0, _ = first_file(dir)
    files = extract_file_list(dir)
    ratio = []
    rt_std = []
    q_p = np.load(file0)['q']
    for num, path in files:
        q = np.load(path)['q']
        ratio.append((q/q_p).mean())
        rt_std.append((q/q_p).std())
        q_p = q
    return np.array(rt_std)/np.array(ratio)

def does_decay(n: int):
    dir = pdir / f'results/hill_steadEcay{n},Ek0.0e+00,Re2e+03,N128'
    time = time_diff(dir)
    plt.loglog(time, std_decay(dir), label=r'$\sigma/\mu$')
    plt.xlabel('t')
    plt.ylabel(r'$q(t)/q(t-dt)$')
    plt.legend()
    plt.show()
    file, _ = first_file(dir)
    plot_vorticity_file(file, f'$q_{{{n}}}(0)$')
    file, _ = last_file(dir)
    plot_vorticity_file(file, f'$q_{{{n}}}(\\infty)$')

def does_decay2(n: int):
    dir = pdir / f'results/hill_steadEcay{n},Ek0.0e+00,Re2e+03,N128'
    time = time_diff(dir)
    plt.plot(time, avg_diff(dir), label=r'$\mu$')
    plt.plot(time, std_diff(dir), label=r'$\sigma$')
    plt.xlabel('t')
    plt.ylabel(r'$q(t)-q(t-dt)$')
    plt.legend()
    plt.show()

def does_decay3(n:int):
    from po2d_config import Dx,N,Re
    import po2d_lib as po
    dir = pdir / f'results/hill_steadEcay{n},Ek0.0e+00,Re2e+03,N128'
    topo = po.gauss_topography(Dx, N)
    files = extract_file_list(dir)
    stat_eq = []
    stat_eq_w = []
    for num, path in files:
        fluid = po.FluidStateTopography(topo, vorticity=np.load(path)['q'])
        fluid.streamfunction(0)
        E = fluid.energy()
        fluid.q /= np.sqrt(E)
        fluid.streamfunction(1)
        fluid.arakawa_jacobian(1)
        lap_z = fluid.dissipation()/Dx**2
        E = fluid.energy()
        E_diss = (fluid.psi*lap_z).sum()/Re
        q_dot = -fluid.J + lap_z/Re
        stat_eq_w.append(np.abs(q_dot).mean())
        q_dot -= fluid.vorticity() * E_diss/E
        stat_eq.append(np.abs(q_dot).mean())
    time = time_diff(dir)
    plt.plot(time, np.array(stat_eq), label='right')
    plt.plot(time, np.array(stat_eq_w), label='wrong')
    plt.xlabel('t')
    plt.ylabel(r'$-J(q,\psi)+\nu\Delta(\zeta)$')
    plt.legend()
    plt.show()
    