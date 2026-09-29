import numpy as np
import cmath
import matplotlib.pyplot as plt
import fns
import scipy
hbar = scipy.constants.hbar

phi = np.pi
exp = cmath.exp(1j*phi)
t_max = 5e-6
del_t = 1e-8
tau = 8e-7
t_s = np.arange(0, t_max, del_t)

del_max = 1e7
rabi_max = 3e6

Q = np.identity(3)


psi_0 = np.array([1,1,0,1,1,0,0,0,0])

P_11 = np.zeros(len(t_s))
P_rr = np.zeros(len(t_s))

fig, axs = plt.subplots(3,1)
print(psi_0)
linewidth = 1
color = 'b'
V_1=0
V_r=0
for i in range (3):
    P = (0j) * np.zeros((len(t_s), 3**2))
    P[0] = psi_0
    P_11[0] = abs(psi_0[4]) ** 2
    P_rr[0] = abs(psi_0[-1])**2
    H_V = np.zeros((3**2, 3**2))
    H_V[-1, -1] = 2*V_r
    H_V[4,4] = 2*V_1

    for i in range(len(t_s)-1):
        H_1 = fns.qutrit_H(fns.sine_del(t_s[i],t_max,del_max), fns.square_rabi(t_s[i],t_max,tau,rabi_max), exp)
        H_2 = np.kron(H_1,Q) + np.kron(Q,H_1) + H_V

        P[i+1] = np.dot(scipy.linalg.expm(-1j * H_2 * del_t), P[i])
        P_11[i+1] = abs(P[i+1][4])**2
        P_rr[i+1] = abs(P[i+1][-1])**2


    axs[1].plot(t_s, P_11, linewidth = linewidth, color = color)
    axs[1].set_xlabel(r'$Time$ / $\mu s$')
    axs[1].set_ylabel(r'$P_{|11\rangle}$')

    axs[2].plot(t_s, P_rr, linewidth=linewidth, color=color)
    axs[2].set_xlabel(r'$Time$ / $\mu s$')
    axs[2].set_ylabel(r'$P_{|rr\rangle}$')
    if V_1==0:
        print(P[200])
        print(P[-1])

    V_1 += 7e5
    V_r += 7e5

    linewidth = 0.1
    color = 'red'

axs[1].set_ylim([0,1])
axs[2].set_ylim([0,1])

rabi_ = [fns.square_rabi(i,t_max,tau,rabi_max)/(2*np.pi)/1e6 for i in t_s]
del_ = [fns.sine_del(i, t_max, del_max)/(2*np.pi)/1e6 for i in t_s]

twin = axs[0].twinx()
axs[0].plot(t_s, rabi_, linewidth=1, color='r', label = '$\Omega/2\pi$')
twin.plot(t_s, del_, linewidth=1, color='b', label = '$\Delta/2\pi$')
axs[0].set_xlabel(r'$Time$ / $\mu s$')
axs[0].set_ylabel(r'MHz')
axs[0].legend(axs[0],twin, loc=0)


fig.tight_layout()
plt.show()