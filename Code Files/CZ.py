import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import host_subplot
import qutrits
import numpy as np
import cmath
import fns
import qutip
import scipy
from scipy import optimize

phi = 0
t_max = 3e-6
del_t = 1e-8
tau = 3e-6
tau_2 = 8.5e-7*3
t_s = np.arange(0, t_max, del_t)
exp = cmath.exp(1j*phi)


del_max = 0
rabi_max = 0.349*2*np.pi*1e6

V=5e12

del_fn = fns.single_del
rabi_fn = fns.square_rabi

psi_0 = np.array([0,0,0,0,1,0,0,0,0])
psi_0 = psi_0/scipy.linalg.norm(psi_0)

l=3
n=2

labels, prob, psi = qutrits.two_qutrit(psi_0,del_fn,rabi_fn,del_max,rabi_max,V,t_max,del_t,tau,exp)

fig, axs = plt.subplots(l**n,1)

X, Y = np.meshgrid(t_s, [0, 1])
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.71])

# for i in range(l ** n):
#     norm = colors.Normalize(vmin=np.min(prob), vmax=np.max(prob))
#
#     Z = [prob[i], prob[i]]
#
#     cm = axs[i].contourf(X, Y, Z, 40, extend="max", norm=norm)
#     axs[i].set_ylabel(labels[i])
#     axs[i].set_ylim([0, 1])
#     axs[i].set_position([.1,.8-i*.08,.7,.08])
#     axs[i].set_yticks([])
#     if i != 8:
#         axs[i].set_xticks([])
#     if i== 1:
#         fig.colorbar(cm, cax=cbar_ax)
# axs[-1].set_xlabel(r'$Time$ / $\mu s$')

def CZ_plot(psi_0, rabi_fn, del_fn, rabi_max, del_max):
    l = 3
    n = 2

    labels, prob, psi = qutrits.two_qutrit(psi_0, del_fn(del_max),del_max, rabi_fn(rabi_max,tau), rabi_max, V, t_max, del_t, tau, exp)

    fig, axs = plt.subplots(l ** n, 1)

    X, Y = np.meshgrid(t_s, [0, 1])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.71])

    for i in range(l ** n):
        norm = colors.Normalize(vmin=np.min(prob), vmax=np.max(prob))

        Z = [prob[i], prob[i]]

        cm = axs[i].contourf(X, Y, Z, 40, extend="max", norm=norm)
        axs[i].set_ylabel(labels[i])
        axs[i].set_ylim([0, 1])
        axs[i].set_position([.1, .8 - i * .08, .7, .08])
        axs[i].set_yticks([])
        if i != 8:
            axs[i].set_xticks([])
        if i == 1:
            fig.colorbar(cm, cax=cbar_ax)
    axs[-1].set_xlabel(r'$Time$ / $\mu s$')

    return fig
CZ = np.matrix(np.array([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],
                           [0,0,0,1,0,0,0,0,0],[0,0,0,0,-1,0,0,0,0],[0,0,0,0,0,1,0,0,0],
                           [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]]))
rho_CZ = CZ @ (np.matrix(psi_0).getH() @ np.matrix(psi_0))  @ CZ.getH()
rabis = np.arange(0e6, 1.8e6, 1e4)
psis = []
psis_r = []
fid = []
# for i in range(len(rabis)):
#     psis.append(qutrits.two_qutrit(psi_0,del_fn,rabi_fn,del_max,rabis[i],V,t_max,del_t,tau,exp)[2])
#     psis_r.append(qutrits.two_qutrit(psi_0, del_fn, rabi_fn, del_max, rabis[i], V, t_max, del_t, tau, exp)[2].real)
#     rho = np.matrix(psis[i]).getH() @ np.matrix(psi)
#     fid.append(qutip.fidelity(qutip.Qobj(rho), qutip.Qobj(rho_CZ)))

plt.figure()
host = host_subplot(111)
par = host.twinx()

host.set_xlabel('$\Omega/2\pi$ (MHz)')
host.set_ylabel('Re($\psi$)')
par.set_ylabel("Fidelity")
labels1 = [r'$|00\rangle$',r'$|01\rangle$',r'$|0r\rangle$',r'$|10\rangle$',r'$|11\rangle$',r'$|1r\rangle$',r'$|r0\rangle$',r'$|r1\rangle$', r'$|rr\rangle$']

# p1 = host.plot(rabis*1e-6/2/np.pi, psis_r, label=labels1)
# p2 = par.plot(rabis*1e-6/(2*np.pi), fid, label='Fidelity', linestyle = ':')

# host.legend(labelcolor="linecolor")

# host.yaxis.label.set_color(p1.get_color())
# par.yaxis.label.set_color(p2.get_color())

# plt.show()


def func(x, a, b, c):
    return a * np.sin(b * (x + c))

# *params1, params_covariance1 = optimize.curve_fit(func, rabis, psis_r, p0=[.5, 1 / 1.385e6 * 2 * np.pi, -2.3e6])
# print(params1)

print(psi)



def CZ_col(psi, del_fn, rabi_fn, del_max, rabi_max, V, tau, c_ops):
    pauli_x = fns.qutrit_H01(0, 1, cmath.exp(1j * 0))
    pauli_y = fns.qutrit_H01(0, 1, cmath.exp(1j * np.pi / 2))
    iden = fns.qutrit_H01(1, 0, cmath.exp(1j * 0))

    del_t = 1e-8

    Q = np.identity(3)
    t_s = np.arange(0, tau, del_t)

    H_V = np.zeros((3 ** 2, 3 ** 2))
    H_V[-1, -1] = V
    H_1 = fns.qutrit_H12(del_fn(t_max, t_max, del_max), rabi_fn(t_max, t_max, tau, rabi_max), exp)
    H_2 = fns.qutrit_H12(del_fn(t_max, t_max, del_max), rabi_fn(t_max, t_max, tau, rabi_max), exp)

    H = np.kron(H_1, Q) + np.kron(Q, H_2) + H_V

    rhos_1 = qutip.mesolve(qutip.Qobj(H), qutip.Qobj(psi), t_s, c_ops=c_ops, options=qutip.Options(nsteps=200000)).states

    return rhos_1

#collapse operators
gam_rs = np.arange(0,1.5e3,200)
gam_p = 6

def r_col(gam):
    return gam * np.array([[0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0],[0,0,0,0,0,1,0,1,0],[0,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0]])
def p_col(gam):
    return gam * np.array([[0,1,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]])
# rho_1 = CZ_col(psi, del_fn, rabi_fn,del_max,rabi_max,1e12,1e-7,c_ops=[])
# rho_2 = CZ_col(psi, del_fn, rabi_fn,del_max,rabi_max,1e12,1e-7,c_ops=qutip.Qobj(p_col))

# print(rho_1[-1])
# fig, axs = plt.subplots(1,3)
# qutip.hinton(rho_1[-1]*rho_1[-1].dag(), ax=axs[0])
# qutip.hinton(rho_2[-1]*rho_2[-1].dag(), ax=axs[1])

# qutip.hinton(qutip.Qobj(psi.real) * qutip.Qobj(psi).dag(), ax =axs[2])

plt.show()

psi_0s = [np.array([1,0,0,0,1,0,0,0,0])/np.sqrt(2),np.array([0,0,0,1,1,0,0,0,0])/np.sqrt(2)]
# ,np.array([0,0,0,0,1,0,0,0,0]),np.array([0,0,1,1,1,0,0,0,0])/np.sqrt(3)
fid_r_ave=[]
fid_p_ave=[]
for i in gam_rs:
    fid_r=[]
    fid_p=[]
    for j in psi_0s:
        rho_r = CZ_col(j, del_fn, rabi_fn,del_max,rabi_max,1e12,tau,c_ops=qutip.Qobj(r_col(i)))[-1]
        # rho_p = CZ_col(j, del_fn, rabi_fn,del_max,rabi_max,1e12,tau,c_ops=qutip.Qobj(p_col(i)))[-1]
        rho_CZ = CZ @ (np.matrix(j).getH() @ np.matrix(j)) @ CZ.getH()

        fid_r.append(qutip.fidelity(qutip.Qobj(rho_r), qutip.Qobj(rho_CZ)))
        # fid_p.append(qutip.fidelity(qutip.Qobj(rho_p), qutip.Qobj(rho_CZ)))
    fid_r_ave.append(np.average(fid_r))
    # fid_p_ave.append(np.average(fid_p))


def func_2(x, a, b, c):
    return a * np.exp(-b*(x)) +c

# *params2, params_covariance2 = optimize.curve_fit(func_2, gam_rs, fid_r_ave, p0=[1, 1/2000, 0.7])
# print(params2)
# print(params2[0])
# model = [func_2(i,params2[0][0],params2[0][1],params2[0][2]) for i in gam_rs]
# print(model)
# st_dev = np.array([model,fid_r_ave]).std(axis=0)
plt.figure()
# plt.plot(gam_rs, fid_p_ave)
plt.scatter((gam_rs**2)/1e6, fid_r_ave, marker='^')
# plt.plot(gam_rs, model , linestyle=':')

plt.xlabel('$\gamma/MHz$')
plt.ylabel('fidelity')
plt.show()