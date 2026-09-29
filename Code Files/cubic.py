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
Rabi = 1e6 *2*np.pi
phi = 0


exp = cmath.exp(1j*phi)
Q = np.identity(2)


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
        H_0 = fns.H(fns.cubic_del(t_s[i], t_max), Rabi, exp)
        H_2 = hbar/2 * (np.kron(H_0, Q) + np.kron(Q, H_0) + H_V)


        P[i+1] = np.dot(scipy.linalg.expm(-1j*H_2*del_t/hbar), P[i])

        P_00.append(abs(P[i+1][0]**2))
        P_psi.append(abs(P[i+1][2]**2 + P[i+1][1]**2))

    axs[1].plot(t_s*1e6, P_00, linewidth=linewidth, color=color)
    axs[1].set_ylim([0, 1])
    axs[2].plot(t_s*1e6, P_psi, linewidth=linewidth, color=color)
    axs[2].set_ylim([0, 1])

    V += 1e6
    linewidth = 0.1
    color = 'red'

del_ = np.array([fns.cubic_del(i, t_max) for i in t_s])
axs[0].plot(t_s, del_*1e-6)

axs[0].set_ylabel(r'$\Delta$ / $MHz$')

axs[1].set_ylabel(r'$P_{|00\rangle}$')

axs[2].set_xlabel(r'$Time$ / $\mu s$')
axs[2].set_ylabel(r'$P_{|\psi^+\rangle}$')

axs[0].set_position([.15,.646,.8,.2])
axs[1].set_position([.15,.423,.8,.2])
axs[2].set_position([.15,.2,.8,.2])

axs[1].set_yticks([0,.5,1])
axs[2].set_yticks([0,.5,1])

axs[0].set_xticks([])
axs[1].set_xticks([])

plt.show()

fig.savefig('Time-dep-H.png')