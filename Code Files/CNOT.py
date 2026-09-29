import matplotlib.pyplot as plt
from matplotlib import colors
import qutrits
import numpy as np
import cmath
import fns
import Hadamard
import qutip
import scipy

phi = 0
t_max = 3e-6
del_t = 1e-8

t_s = np.arange(0, t_max, del_t)
exp = cmath.exp(1j*phi)

CZdel_max = 0
CZrabi_max = 0.3482 * 1e6 * 2*np.pi
CZtau = 3e-6

Haddel_max = 0
Hadrabi_1 = 1/2 * np.pi *1e6
Hadrabi_2 = 1/4 * np.pi *1e6
Hadtau = 1e-6

V=1e12

CZdel_fn = fns.single_del
CZrabi_fn = fns.square_rabi

Haddel_fn = fns.single_del
HadRabi_fn = fns.square_rabi

psi_0 = np.array([1,0,0,1,0,0,0,0,0])
psi_0 = psi_0/scipy.linalg.norm(psi_0)

l=3
n=2


prob_1, psi_1, t_s1 = Hadamard.Hadamard_trit(psi_0,Haddel_fn,HadRabi_fn, Haddel_max, Hadrabi_1,Hadrabi_2,Hadtau)
labels, prob_2, psi_2 = qutrits.two_qutrit(psi_1, CZdel_fn, CZrabi_fn,CZdel_max,CZrabi_max, V, t_max, del_t, CZtau, exp)
prob_3, psi_3, t_s3 = Hadamard.Hadamard_trit(psi_2,Haddel_fn,HadRabi_fn, Haddel_max, Hadrabi_1,Hadrabi_2,Hadtau)


prob = np.concatenate((prob_1, prob_2, prob_3), axis=1)
fig, axs = plt.subplots(l**n, 1)

t_2 = np.array(t_s) + t_s1[-1]
t_3 = np.array(t_s3) + t_2[-1]
t=np.concatenate((t_s1 ,t_2, t_3))

X, Y = np.meshgrid(t*1e6, [0, 1])
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.71])

for i in range(l ** n):
    norm = colors.Normalize(vmin=np.min(prob), vmax=np.max(prob))

    Z = [prob[i], prob[i]]

    cm = axs[i].contourf(X, Y, Z, 40, extend="max", norm=norm)
    axs[i].set_ylabel(labels[i])
    axs[i].set_ylim([0, 1])
    axs[i].set_position([.1,.8-i*.08,.7,.08])
    axs[i].set_yticks([])
    if i != 8:
        axs[i].set_xticks([])
    if i== 1:
        fig.colorbar(cm, cax=cbar_ax)
print(psi_3)

rho_0 = np.matrix(psi_0).getH() @ np.matrix(psi_0)
rho_1 = np.matrix(psi_1).getH() @ np.matrix(psi_1)
rho_2 = np.matrix(psi_2).getH() @ np.matrix(psi_2)
rho_3 = np.matrix(psi_3).getH() @ np.matrix(psi_3)

CNOT = np.matrix(np.array([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],
                           [0,0,0,0,1,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],
                           [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]]))

rho_CNOT = CNOT @ (np.matrix(psi_0).getH() @ np.matrix(psi_0))  @ CNOT.getH()

print(qutip.fidelity(qutip.Qobj(rho_3),qutip.Qobj(rho_CNOT)))


labelsy = [r'$|00\rangle$',r'$|01\rangle$', r'$|0r\rangle$', r'$|10\rangle$', r'$|11\rangle$',r'$|1r\rangle$', r'$|r0\rangle$', r'$|r1\rangle$', r'$|rr\rangle$']
labelsx = [r'$\langle 00|$',r'$\langle 01|$', r'$\langle 0r|$', r'$\langle 10|$',r'$\langle 11|$', r'$\langle 1r|$',r'$\langle r0|$',r'$\langle r1|$', r'$\langle rr|$']

axs[-1].set_xlabel(r'$Time$ / $\mu s$')
qutip.hinton(qutip.Qobj(rho_0),x_basis = labelsx, y_basis=labelsy, cmap=None)
qutip.hinton(qutip.Qobj(rho_1),x_basis = labelsx, y_basis=labelsy, cmap=None)
qutip.hinton(qutip.Qobj(rho_2),x_basis = labelsx, y_basis=labelsy, cmap=None)
qutip.hinton(qutip.Qobj(rho_3),x_basis = labelsx, y_basis=labelsy)

print(t[-1])
plt.show()