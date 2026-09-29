import numpy as np
import cmath
import matplotlib.pyplot as plt
import fns
import qutrits
import qutip
from qutip import *
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import constants
hbar = scipy.constants.hbar


phi = 0
t_max = 5e-5
del_t = 1e-8
tau = 8.5e-7/2
tau_2 = t_max
t_s = np.arange(0, t_max, del_t)
exp = cmath.exp(1j*phi)

del_max = 1e6
rabi_max = 2.887e6

psi_0 = np.array([1,1,1,0,1,0,0,0,0])
psi_0 = psi_0/np.sqrt(4)

V=2.5e6 *2

def density(psi_0,del_max,rabi_max,V,t_max,del_t,tau,exp):

    Q = np.identity(3)

    P = (0j) * np.zeros((len(t_s), 3**2))
    P[0] = psi_0

    rho = (0j) * np.zeros((len(t_s), 3**2, 3**2))
    rho[0] = np.matrix(P[0]).getH() @ np.matrix(P[0])

    Prob_t = np.zeros((3**2, len(t_s)))
    for i in range (3**2):
        Prob_t[i][0] = abs(P[0][i])**2

    H_V = np.zeros((3**2, 3**2))
    H_V[-1, -1] = V

    for i in range(len(t_s)-1):
        H_1 = fns.qutrit_H12(fns.sine_del(t_s[i], t_max, del_max), fns.square_rabi(t_s[i],t_max,tau,rabi_max), exp)
        H_2 = fns.qutrit_H12(fns.sine_del(t_s[i], t_max, del_max), fns.square_rabi(t_s[i],t_max,tau,rabi_max), exp)

        H = np.kron(H_1,Q) + np.kron(Q,H_2) + H_V

        rho[i+1] = np.matrix(scipy.linalg.expm(-1j * H * del_t)) @ rho[i] @ np.matrix(scipy.linalg.expm(-1j * H * del_t)).getH()




    labels = [r'$P_{|00\rangle}$',r'$P_{|01\rangle}$', r'$P_{|0r\rangle}$', r'$P_{|10\rangle}$', r'$P_{|11\rangle}$',r'$P_{|1r\rangle}$', r'$P_{|r0\rangle}$', r'$P_{|r1\rangle}$', r'$P_{|rr\rangle}$']


    return rho


def decoherence(psi_0,del_max,rabi_max,V,t_max,del_t,tau,exp):
    pauli_x = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    pauli_y = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    pauli_z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

    Q = np.identity(3)

    P = (0j) * np.zeros((len(t_s), 3**2))
    P[0] = psi_0

    rho = (0j) * np.zeros((len(t_s), 3**2, 3**2))
    rho[0] = np.matrix(P[0]).getH() @ np.matrix(P[0])

    Prob_t = np.zeros((3**2, len(t_s)))
    for i in range (3**2):
        Prob_t[i][0] = abs(P[0][i])**2

    H_V = np.zeros((3**2, 3**2))
    H_V[-1, -1] = V

    for i in range(len(t_s)-1):
        H_1 = fns.qutrit_H12(fns.sine_del(t_s[i], t_max, del_max), fns.square_rabi(t_s[i],t_max,tau,rabi_max), exp)
        H_2 = fns.qutrit_H12(fns.sine_del(t_s[i], t_max, del_max), fns.square_rabi(t_s[i],t_max,tau,rabi_max), exp)

        H = np.kron(H_1,Q) + np.kron(Q,H_2) + H_V

    result = qutip.mesolve(qutip.Qobj(H), qutip.Qobj(rho[0]), t_s, [np.sqrt(0.05) * qutip.Qobj(np.kron(pauli_x,np.identity(3)))], [])

    rho[i+1] = np.matrix(scipy.linalg.expm(-1j * H * del_t)) @ rho[i] @ np.matrix(scipy.linalg.expm(-1j * H * del_t)).getH()




    labels = [r'$P_{|00\rangle}$',r'$P_{|01\rangle}$', r'$P_{|0r\rangle}$', r'$P_{|10\rangle}$', r'$P_{|11\rangle}$',r'$P_{|1r\rangle}$', r'$P_{|r0\rangle}$', r'$P_{|r1\rangle}$', r'$P_{|rr\rangle}$']


    return rho, result

def decoherence_ind(psi_0,del_max,rabi_max,V,t_max,del_t,tau,exp):
    pauli_x = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    pauli_y = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
    pauli_z = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

    Q = np.identity(3)

    P = (0j) * np.zeros((len(t_s), 3**2))
    P[0] = psi_0

    rho = (0j) * np.zeros((len(t_s), 3**2, 3**2))
    rho[0] = np.matrix(P[0]).getH() @ np.matrix(P[0])

    Prob_t = np.zeros((3**2, len(t_s)))
    for i in range (3**2):
        Prob_t[i][0] = abs(P[0][i])**2

    H_V = np.zeros((3**2, 3**2))
    H_V[-1, -1] = V


    H_1 = fns.qutrit_H12(del_max, rabi_max, exp)
    H_2 = fns.qutrit_H12(del_max, rabi_max, exp)

    H = np.kron(H_1,Q) + np.kron(Q,H_2) + H_V

    result = qutip.mesolve(qutip.Qobj(H), qutip.Qobj(psi_0), t_s, [], [])

    rho[i+1] = np.matrix(scipy.linalg.expm(-1j * H * del_t)) @ rho[i] @ np.matrix(scipy.linalg.expm(-1j * H * del_t)).getH()




    labels = [r'$P_{|00\rangle}$',r'$P_{|01\rangle}$', r'$P_{|0r\rangle}$', r'$P_{|10\rangle}$', r'$P_{|11\rangle}$',r'$P_{|1r\rangle}$', r'$P_{|r0\rangle}$', r'$P_{|r1\rangle}$', r'$P_{|rr\rangle}$']


    return rho, result
l=3
n=2

print(density(psi_0,del_max,rabi_max,V,t_max,del_t,tau,exp)[1])
rho_t = density(psi_0,del_max,rabi_max,V,t_max,del_t,tau,exp)
result = decoherence_ind(psi_0,del_max,rabi_max,V,t_max,del_t,tau,exp)[1]


# fig = plt.figure()

# qutip.visualization.plot_fock_distribution(result.states[-1])

# plt.plot(result.times,result.states)

psi = np.array([abs(i.full()) for i in result.states])
psi = psi[0]
print(np.shape(psi))
plt.plot(result.times, psi[:,:][0])


# qutip.plot_expectation_values(result.states.real)
# qutip.hinton(psi[-1].imag)

# b = qutip.Bloch()
#
# b.add_points(result.states[:][0])
plt.show()
