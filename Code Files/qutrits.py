import numpy as np
import cmath
import matplotlib.pyplot as plt
import fns
import scipy
from scipy import constants
hbar = scipy.constants.hbar


t_max = 8e-7 * 4
del_t = 1e-8
tau = 8e-7
t_s = np.arange(0, t_max, del_t)

del_max = 1e6
rabi_max = 2.887e6

def two_qutrit(psi_0,del_fn,rabi_fn,del_max,rabi_max,V,t_max,del_t,tau,exp):

    Q = np.identity(3)
    t_s = np.arange(0, t_max, del_t)

    P = (0j) * np.zeros((len(t_s), 3**2))
    P[0] = psi_0

    Prob_t = np.zeros((3**2, len(t_s)))
    for i in range (3**2):
        Prob_t[i][0] = abs(P[0][i])**2

    H_V = np.zeros((3**2, 3**2))
    H_V[-1, -1] = V

    for i in range(len(t_s)-1):
        H_1 = fns.qutrit_H12(del_fn(t_s[i], t_max, del_max), rabi_fn(t_s[i],t_max,tau,rabi_max), exp)
        H_2 = fns.qutrit_H12(del_fn(t_s[i], t_max, del_max), rabi_fn(t_s[i],t_max,tau,rabi_max), exp)

        H = np.kron(H_1,Q) + np.kron(Q,H_2) + H_V

        P[i+1] = np.dot(scipy.linalg.expm(-1j * H * del_t), P[i])

        for n in range (3**2):
            Prob_t[n][i+1] = abs(P[i+1][n])**2

    labels = [r'$|00\rangle$', r'$|01\rangle$', r'$|0r\rangle$', r'$|10\rangle$', r'$|11\rangle$', r'$|1r\rangle$',
               r'$|r0\rangle$', r'$|r1\rangle$', r'$|rr\rangle$']

    return labels, Prob_t, P[-1]

def two_qutritind(psi_0,del_fn,rabi_fn,del_max,rabi_max,V,t_max,del_t,tau,exp):

    Q = np.identity(3)

    Prob_t = np.zeros((3**2))

    H_V = np.zeros((3**2, 3**2))
    H_V[-1, -1] = V


    H_1 = fns.qutrit_H12(del_fn(t_max, t_max, del_max), rabi_fn(t_max,t_max,tau,rabi_max), exp)
    H_2 = fns.qutrit_H12(del_fn(t_max, t_max, del_max), rabi_fn(t_max,t_max,tau,rabi_max), exp)

    H = np.kron(H_1,Q) + np.kron(Q,H_2) + H_V

    P = np.dot(scipy.linalg.expm(-1j * H * t_max), psi_0)

    for n in range (3**2):
        Prob_t[n] = abs(P[n])**2



    labels = [r'$P_{|00\rangle}$',r'$P_{|01\rangle}$', r'$P_{|0r\rangle}$', r'$P_{|10\rangle}$', r'$P_{|11\rangle}$',r'$P_{|1r\rangle}$', r'$P_{|r0\rangle}$', r'$P_{|r1\rangle}$', r'$P_{|rr\rangle}$']


    return labels, Prob_t, P[-1]

def qutrit_02(psi_0,del_max,rabi_max,V,t_max,del_t,tau,exp):
    fig, axs = plt.subplots(3 + 1, 1)

    P = (0j) * np.zeros((len(t_s), 3))
    P[0] = psi_0

    Prob_t = np.zeros((3, len(t_s)))
    for i in range(3):
        Prob_t[i][0] = abs(P[0][i]) ** 2

    H_V = np.zeros((3, 3))
    H_V[-1, -1] = V

    for i in range(len(t_s) - 1):
        H_1 = fns.qutrit_H02(fns.sine_del(t_s[i], t_max, del_max), fns.square_rabi(t_s[i], t_max, tau, rabi_max), exp)

        P[i + 1] = np.dot(scipy.linalg.expm(-1j * H_1 * del_t), P[i])

        for n in range(3):
            Prob_t[n][i + 1] = abs(P[i+1][n]) ** 2



    labels = [r'$P_{|0\rangle}$', r'$P_{|1\rangle}$', r'$P_{|r\rangle}$']
    for i in range(3):
        X, Y = np.meshgrid(t_s, [0, 1])

        Z = [Prob_t[i], Prob_t[i]]

        cm = axs[i + 1].contourf(X, Y, Z, 30)
        axs[i + 1].set_xlabel(r'$Time$ / $\mu s$')
        axs[i + 1].set_ylabel(labels[i])
        axs[i + 1].set_ylim([0, 1])

    plt.colorbar(cm)

    rabi_ = [fns.square_rabi(i, t_max, tau, rabi_max) / (2 * np.pi) / 1e6 for i in t_s]
    del_ = [fns.sine_del(i, t_max, del_max) / (2 * np.pi) / 1e6 for i in t_s]

    twin = axs[0].twinx()
    axs[0].plot(t_s, rabi_, linewidth=1, color='r', label='$\Omega/2\pi$')
    twin.plot(t_s, del_, linewidth=1, color='b', label='$\Delta/2\pi$')
    axs[0].set_xlabel(r'$Time$ / $\mu s$')
    axs[0].set_ylabel(r'MHz')

    fig.tight_layout()

def qutrit_01(psi_0,del_max,rabi_max,V,t_max,del_t,tau,exp):
    fig, axs = plt.subplots(3 + 1, 1)

    P = (0j) * np.zeros((len(t_s), 3))
    P[0] = psi_0

    Prob_t = np.zeros((3, len(t_s)))
    for i in range(3):
        Prob_t[i][0] = abs(P[0][i]) ** 2

    H_V = np.zeros((3, 3))
    H_V[-1, -1] = V

    for i in range(len(t_s) - 1):
        H_1 = fns.qutrit_H01(fns.sine_del(t_s[i], t_max, del_max), fns.square_rabi(t_s[i], t_max, tau, rabi_max), exp)

        P[i + 1] = np.dot(scipy.linalg.expm(-1j * H_1 * del_t), P[i])

        for n in range(3):
            Prob_t[n][i + 1] = abs(P[i+1][n]) ** 2



    labels = [r'$P_{|0\rangle}$', r'$P_{|1\rangle}$', r'$P_{|r\rangle}$']
    for i in range(3):
        X, Y = np.meshgrid(t_s, [0, 1])

        Z = [Prob_t[i], Prob_t[i]]

        cm = axs[i + 1].contourf(X, Y, Z, 30)
        axs[i + 1].set_xlabel(r'$Time$ / $\mu s$')
        axs[i + 1].set_ylabel(labels[i])
        axs[i + 1].set_ylim([0, 1])
    plt.colorbar(cm)


    rabi_ = [fns.square_rabi(i, t_max, tau, rabi_max) / (2 * np.pi) / 1e6 for i in t_s]
    del_ = [fns.sine_del(i, t_max, del_max) / (2 * np.pi) / 1e6 for i in t_s]

    twin = axs[0].twinx()
    axs[0].plot(t_s, rabi_, linewidth=1, color='r', label='$\Omega/2\pi$')
    twin.plot(t_s, del_, linewidth=1, color='b', label='$\Delta/2\pi$')
    axs[0].set_xlabel(r'$Time$ / $\mu s$')
    axs[0].set_ylabel(r'MHz')


    fig.tight_layout()
def qutrit_12(psi_0,del_max,rabi_max,V,t_max,del_t,tau,exp):
    fig, axs = plt.subplots(3 + 1, 1)

    P = (0j) * np.zeros((len(t_s), 3))
    P[0] = psi_0

    Prob_t = np.zeros((3, len(t_s)))
    for i in range(3):
        Prob_t[i][0] = abs(P[0][i]) ** 2



    for i in range(len(t_s) - 1):
        H_1 = fns.qutrit_H12(fns.sine_del(t_s[i], t_max, del_max), fns.square_rabi(t_s[i], t_max, tau, rabi_max), exp)

        P[i + 1] = np.dot(scipy.linalg.expm(-1j * H_1 * del_t), P[i])

        for n in range(3):
            Prob_t[n][i + 1] = abs(P[i+1][n]) ** 2



    labels = [r'$P_{|0\rangle}$', r'$P_{|1\rangle}$', r'$P_{|r\rangle}$']

    for i in range(3):
        X,Y = np.meshgrid(t_s,[0,1])

        Z = [Prob_t[i], Prob_t[i]]

        cm = axs[i + 1].contourf(X,Y,Z,30)
        axs[i + 1].set_xlabel(r'$Time$ / $\mu s$')
        axs[i + 1].set_ylabel(labels[i])
        axs[i + 1].set_ylim([0,1])
    plt.colorbar(cm)


    rabi_ = [fns.square_rabi(i, t_max, tau, rabi_max) / (2 * np.pi) / 1e6 for i in t_s]
    del_ = [fns.sine_del(i, t_max, del_max) / (2 * np.pi) / 1e6 for i in t_s]

    twin = axs[0].twinx()
    axs[0].plot(t_s, rabi_, linewidth=1, color='r', label='$\Omega/2\pi$')
    twin.plot(t_s, del_, linewidth=1, color='b', label='$\Delta/2\pi$')
    axs[0].set_xlabel(r'$Time$ / $\mu s$')
    axs[0].set_ylabel(r'MHz')


    fig.tight_layout()

def interaction(psi_0,del_max,rabi_max,V,t_max,del_t,tau,exp):

    Q = np.identity(3)

    P = (0j) * np.zeros((len(t_s), 3**2))
    P[0] = psi_0

    Prob_t = np.zeros((3**2, len(t_s)))
    for i in range (3**2):
        Prob_t[i][0] = abs(P[0][i])**2

    H_V = np.zeros((3**2, 3**2))
    H_V[-1, -1] = V

    for i in range(len(t_s)-1):
        H_1 = fns.qutrit_H12(fns.sine_del(t_s[i], t_max, del_max), fns.square_rabi(t_s[i],t_max,tau,rabi_max), exp)
        H_2 = fns.qutrit_H12(fns.sine_del(t_s[i], t_max, del_max), fns.square_rabi(t_s[i],t_max,tau,rabi_max), exp)

        H = np.kron(H_1,Q) + np.kron(Q,H_2) + H_V

        P[i+1] = np.dot(scipy.linalg.expm(-1j * H * del_t), P[i])

        for n in range (3**2):
            Prob_t[n][i+1] = abs(P[i+1][n])**2



    labels = [r'$P_{|00\rangle}$',r'$P_{|01\rangle}$', r'$P_{|0r\rangle}$', r'$P_{|10\rangle}$', r'$P_{|11\rangle}$',r'$P_{|1r\rangle}$', r'$P_{|r0\rangle}$', r'$P_{|r1\rangle}$', r'$P_{|rr\rangle}$']


    return labels, Prob_t, P[-1]