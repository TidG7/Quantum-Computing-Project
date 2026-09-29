import numpy as np
import cmath
import matplotlib.pyplot as plt
import scipy
import fns
hbar = scipy.constants.hbar


# set variables
t_max = 1e-6
psi_0 = np.array([1,0,0,0])
del_t = 5e-9
t_s = np.arange(0, t_max, del_t)

phi = 0

tau = 1e-7
rabi_max = 1.6e7

del_max = 3e7

Q = np.identity(2)
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
    H_V = np.zeros((2 ** 2, 2 ** 2))
    H_V[-1, -1] = 2 * V

    for i in range(len(t_s) - 1):
        H_0 = fns.H(fns.sine_del(t_s[i], t_max, del_max), fns.adiabatic_rabi(t_s[i], t_max, tau, rabi_max), exp)
        H_2 = hbar/2 * (np.kron(H_0, Q) + np.kron(Q, H_0) + H_V)
        P[i+1] = np.dot(scipy.linalg.expm(-1j*H_2*del_t/hbar), P[i])
        P_00.append(abs(P[i+1][0]**2))
        P_psi.append(abs(P[i+1][-1])**2)

    axs[1].plot(t_s, P_00, linewidth=linewidth, color=color)
    axs[1].set_ylim([0, 1])
    axs[1].set_xlabel(r'$Time$ / $\mu s$')
    axs[1].set_ylabel(r'$P_{|11\rangle}$')
    axs[2].plot(t_s, P_psi, linewidth=linewidth, color=color)
    axs[2].set_ylim([0, 1])
    axs[2].set_xlabel(r'$Time$ / $\mu s$')
    axs[2].set_ylabel(r'$P_{|rr\rangle}$')

    V += 1e6
    linewidth = 0.1
    color = 'red'

fig.tight_layout()
del_ = [fns.sine_del(i, t_max, del_max) for i in t_s]
rabi_ = np.array([fns.adiabatic_rabi(i, t_max, tau, rabi_max) for i in t_s])
axs[0].plot(t_s, del_)
axs[0].plot(t_s, rabi_, color='r')
plt.show()