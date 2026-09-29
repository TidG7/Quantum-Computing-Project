import numpy as np
import cmath
import matplotlib.pyplot as plt
import time_evolve
import fns
import scipy
hbar = scipy.constants.hbar

# set variables
t_max = 1e-8
psi_0 = np.array([1,0,0,0])
del_t = 5e-10
t_s = np.arange(0, t_max, del_t)
Rabi = 1e6 * 2*np.pi
phi = 0

psi_0 = psi_0 * 1 /(sum(psi_0**2)**(1/2))
exp = cmath.exp(1j*phi)

V = 0
linewidth = 1
color = 'blue'
fig, axs = plt.subplots(3)
for i in range (100):
    P = (0j) * np.zeros((len(t_s), 4))
    P[0] = psi_0

    P_00 = [abs(psi_0[0])**2]
    P_psi = [1/np.sqrt(2) * (abs(psi_0[1])**2 + abs(psi_0[2])**2)]

    for i in range(len(t_s) - 1):
        H = hbar/2 * time_evolve.H(detun_fn.linear(t_s[i], t_max), V, Rabi, exp)
        P[i+1] = P[i] - 1j*del_t/hbar * time_evolve.psi(P[i], H, del_t)[0]

        P_00.append(abs(P[i+1][0]**2))
        P_psi.append(1/np.sqrt(2) * (abs(P[i+1][2]**2) + abs(P[i+1][1]**2)) + abs(P[i+1][3]**2))

    axs[1].plot(t_s, P_00, linewidth=linewidth, color=color)
    axs[1].set_xlabel(r'$Time$ / $\mu s$')
    axs[1].set_ylabel(r'$P_{|00\rangle}$')

    axs[2].plot(t_s, P_psi, linewidth=linewidth, color=color)
    axs[2].set_xlabel(r'$Time$ / $\mu s$')
    axs[2].set_ylabel(r'$P_{|\psi^+\rangle}$')

    V += 1e6 * 2*np.pi
    linewidth = 0.1
    color = 'red'

fig.tight_layout()
del_ = []
del_ = [detun_fn.linear(i, t_max) for i in t_s]
axs[0].plot(t_s, del_)
axs[0].set_xlabel(r'$Time$ / $\mu s$')
axs[0].set_ylabel(r'$\Delta$ / $2 \pi$ (MHz) ')

plt.show()
fig.savefig('Time-dep-H.png')