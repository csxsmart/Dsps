import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
plt.rcParams['axes.unicode_minus'] = False  


g = 10.0
L = 1.0
m = 1.0
omega0 = np.sqrt(g / L)


def euler(th, w, h):
    th1 = th + h * w
    w1 = w - h * omega0**2 * np.sin(th)
    return th1, w1


def rk4(th, w, h):
    k1_t = w
    k1_w = -omega0**2 * np.sin(th)

    k2_t = w + 0.5 * h * k1_w
    k2_w = -omega0**2 * np.sin(th + 0.5 * h * k1_t)

    k3_t = w + 0.5 * h * k2_w
    k3_w = -omega0**2 * np.sin(th + 0.5 * h * k2_t)

    k4_t = w + h * k3_w
    k4_w = -omega0**2 * np.sin(th + h * k3_t)

    th1 = th + h * (k1_t + 2*k2_t + 2*k3_t + k4_t) / 6
    w1 = w + h * (k1_w + 2*k2_w + 2*k3_w + k4_w) / 6
    return th1, w1


def symp(th, w, h):
    th1 = th + h * w
    w1 = w - h * omega0**2 * np.sin(th1)
    return th1, w1

def energy(th, w):
    T = 0.5 * m * L**2 * w**2
    V = -m * g * L * np.cos(th)
    return T + V

def sim(step_fn, th0, w0, h, T):
    n = int(T / h)
    t = np.zeros(n + 1)
    th = np.zeros(n + 1)
    w = np.zeros(n + 1)
    E = np.zeros(n + 1)

    th[0] = th0
    w[0] = w0
    E[0] = energy(th0, w0)

    for i in range(n):
        th[i+1], w[i+1] = step_fn(th[i], w[i], h)
        t[i+1] = t[i] + h
        E[i+1] = energy(th[i+1], w[i+1])
    return t, th, w, E

def exp1():
    h = 0.01
    T = 1000.0

    th0_small = 0.5
    th0_large = 2.0
    w0 = 0.0

    methods = {'Euler': euler, 'RK4': rk4, 'Symp': symp}
    cols = {'Euler':'red','RK4':'blue','Symp':'green'}

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    for name, fn in methods.items():
        t, th, w, E = sim(fn, th0_small, w0, h, T)
        Erel = (E - E[0]) / np.abs(E[0])
        plt.plot(t, Erel, label=name, color=cols[name], linewidth=1.5)
    plt.xlabel('t (s)')
    plt.ylabel('δE')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    for name, fn in methods.items():
        t, th, w, E = sim(fn, th0_large, w0, h, T)
        Erel = (E - E[0]) / np.abs(E[0])
        plt.plot(t, Erel, label=name, color=cols[name], linewidth=1.5)
    plt.xlabel('t (s)')
    plt.ylabel('δE')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig1_energy.png', dpi=300, bbox_inches='tight')
    plt.show()


    plt.figure(figsize=(15,5))
    for i,(name,fn) in enumerate(methods.items(),1):
        plt.subplot(1,3,i)
        t, th, w, E = sim(fn, th0_small, w0, h, T)
        plt.plot(th, w, color=cols[name], linewidth=1, alpha=0.7)
        plt.xlabel('th')
        plt.ylabel('w')
        plt.title(name)
        plt.grid(alpha=0.3)
        plt.axis('equal')
    plt.tight_layout()
    plt.savefig('fig2_phase_small.png', dpi=300, bbox_inches='tight')
    plt.show()


    plt.figure(figsize=(15,5))
    for i,(name,fn) in enumerate(methods.items(),1):
        plt.subplot(1,3,i)
        t, th, w, E = sim(fn, th0_large, w0, h, T)
        plt.plot(th, w, color=cols[name], linewidth=1, alpha=0.7)
        plt.xlabel('th')
        plt.ylabel('w')
        plt.title(name)
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig3_phase_large.png', dpi=300, bbox_inches='tight')
    plt.show()


    plt.figure(figsize=(10,6))
    for name, fn in methods.items():
        t, th, w, E = sim(fn, th0_large, w0, h, T)
        Err = np.abs(E - E[0]) / np.abs(E[0])
        plt.semilogy(t, Err, label=name, color=cols[name], linewidth=2)
    plt.xlabel('t (s)')
    plt.ylabel('log(δE)')
    plt.legend()
    plt.grid(which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig4_error.png', dpi=300, bbox_inches='tight')
    plt.show()

def exp2():
    th0 = 2.0
    w0 = 0.0
    T = 500.0
    hs = [0.001, 0.01, 0.05, 0.1]

    methods = {'Euler': euler, 'RK4': rk4, 'Symp': symp}
    cols = {'Euler':'red','RK4':'blue','Symp':'green'}

    plt.figure(figsize=(12,8))
    for i,h in enumerate(hs,1):
        plt.subplot(2,2,i)
        for name,fn in methods.items():
            try:
                t, th, w, E = sim(fn, th0, w0, h, T)
                Erel = (E - E[0]) / np.abs(E[0])
                if np.max(np.abs(Erel)) > 100:
                    plt.plot([],[],label=f"{name} (div)", color=cols[name])
                else:
                    plt.plot(t, Erel, label=name, color=cols[name], linewidth=1.5)
            except Exception:
                plt.plot([],[],label=f"{name} (err)", color=cols[name])
        plt.title(f'h={h}')
        plt.xlabel('t')
        plt.ylabel('δE')
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.ylim([-0.5,2.0])
    plt.tight_layout()
    plt.savefig('fig5_steps.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10,6))
    max_err = {k:[] for k in methods.keys()}
    val_h = {k:[] for k in methods.keys()}
    for h in hs:
        for name,fn in methods.items():
            try:
                t, th, w, E = sim(fn, th0, w0, h, T)
                err = np.max(np.abs((E - E[0]) / E[0]))
                if err < 100:
                    max_err[name].append(err)
                    val_h[name].append(h)
            except Exception:
                pass
    for name in methods.keys():
        if val_h[name]:
            plt.loglog(val_h[name], max_err[name], 'o-', label=name, color=cols[name])
    plt.xlabel('h')
    plt.ylabel('Max(δE)')
    plt.legend()
    plt.grid(which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig6_err_vs_h.png', dpi=300, bbox_inches='tight')
    plt.show()


def driven_rk4(th, w, t0, h, gam, F, om):
    def f(th, w, t):
        return w, -2*gam*w - omega0**2*np.sin(th) + F*np.cos(om*t)
    k1t, k1w = f(th, w, t0)
    k2t, k2w = f(th + 0.5*h*k1t, w + 0.5*h*k1w, t0 + 0.5*h)
    k3t, k3w = f(th + 0.5*h*k2t, w + 0.5*h*k2w, t0 + 0.5*h)
    k4t, k4w = f(th + h*k3t, w + h*k3w, t0 + h)
    th1 = th + h*(k1t + 2*k2t + 2*k3t + k4t)/6
    w1 = w + h*(k1w + 2*k2w + 2*k3w + k4w)/6
    return th1, w1

def driven_symp(th, w, t0, h, gam, F, om):
    th1 = th + h*w
    w1 = w + h*(-2*gam*w - omega0**2*np.sin(th1) + F*np.cos(om*(t0+h)))
    return th1, w1


def sim_driven(method, th0, w0, h, T, gam, F, om):
    n = int(T / h)
    t = np.zeros(n + 1)
    th = np.zeros(n + 1)
    w = np.zeros(n + 1)
    th[0] = th0
    w[0] = w0
    for i in range(n):
        if method == 'RK4':
            th[i+1], w[i+1] = driven_rk4(th[i], w[i], t[i], h, gam, F, om)
        else:
            th[i+1], w[i+1] = driven_symp(th[i], w[i], t[i], h, gam, F, om)
        t[i+1] = t[i] + h
        th[i+1] = np.mod(th[i+1] + np.pi, 2*np.pi) - np.pi
    return t, th, w


def exp3():
    gam = 0.5
    F = 1.5
    om = 2/3
    th0 = 0.2
    w0 = 0.0
    h = 0.01
    Ttrans = 500.0
    T = 2000.0

    t_r, th_r, w_r = sim_driven('RK4', th0, w0, h, T, gam, F, om)
    t_s, th_s, w_s = sim_driven('Symp', th0, w0, h, T, gam, F, om)

    istart = int(Ttrans / h)


    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(th_r[istart::5], w_r[istart::5], ',', alpha=0.5, markersize=0.5)
    plt.xlabel('th')
    plt.ylabel('w')
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    plt.plot(th_s[istart::5], w_s[istart::5], ',', alpha=0.5, markersize=0.5)
    plt.xlabel('th')
    plt.ylabel('w')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig7_attractor.png', dpi=300, bbox_inches='tight')
    plt.show()


    thmin, thmax = -0.5, 0.5
    wmin, wmax = 1.0, 2.0
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    mask = (th_r[istart:] > thmin) & (th_r[istart:] < thmax) & (w_r[istart:] > wmin) & (w_r[istart:] < wmax)
    plt.plot(th_r[istart:][mask], w_r[istart:][mask], 'o', alpha=0.3, markersize=1)
    plt.xlim(thmin, thmax)
    plt.ylim(wmin, wmax)
    plt.grid(alpha=0.3)

    plt.subplot(1,2,2)
    mask2 = (th_s[istart:] > thmin) & (th_s[istart:] < thmax) & (w_s[istart:] > wmin) & (w_s[istart:] < wmax)
    plt.plot(th_s[istart:][mask2], w_s[istart:][mask2], 'o', alpha=0.3, markersize=1)
    plt.xlim(thmin, thmax)
    plt.ylim(wmin, wmax)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig8_zoom.png', dpi=300, bbox_inches='tight')
    plt.show()

    Tper = 2*np.pi/om
    nper = int((T - Ttrans) / (Tper * h))
    p_th_r, p_w_r, p_th_s, p_w_s = [], [], [], []
    for n in range(nper):
        idx = istart + int(n * Tper / h)
        if idx < len(th_r):
            p_th_r.append(th_r[idx]); p_w_r.append(w_r[idx])
            p_th_s.append(th_s[idx]); p_w_s.append(w_s[idx])
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.plot(p_th_r, p_w_r, 'o', markersize=2, alpha=0.6)
    plt.grid(alpha=0.3)
    plt.subplot(1,2,2)
    plt.plot(p_th_s, p_w_s, 'o', markersize=2, alpha=0.6)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig9_poincare.png', dpi=300, bbox_inches='tight')
    plt.show()

    i0 = int(1500 / h)
    i1 = int(1700 / h)
    plt.figure(figsize=(14,5))
    plt.plot(t_r[i0:i1], th_r[i0:i1], label='RK4')
    plt.plot(t_s[i0:i1], th_s[i0:i1], label='Symp', linestyle='--')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig10_ts.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.hist(th_r[istart:], bins=100, density=True, alpha=0.5, label='RK4')
    plt.hist(th_s[istart:], bins=100, density=True, alpha=0.5, label='Symp')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig11_pdf.png', dpi=300, bbox_inches='tight')
    plt.show()

    fs = 1/h
    f_r, P_r = signal.welch(th_r[istart:], fs, nperseg=4096, scaling='density')
    f_s, P_s = signal.welch(th_s[istart:], fs, nperseg=4096, scaling='density')
    plt.figure(figsize=(10,6))
    plt.semilogy(f_r, P_r, label='RK4')
    plt.semilogy(f_s, P_s, label='Symp')
    plt.axvline(om/(2*np.pi), linestyle='--', alpha=0.7)
    plt.xlim([0,5])
    plt.legend()
    plt.grid(which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('fig12_pow.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    exp1()
    exp2()
    exp3()
