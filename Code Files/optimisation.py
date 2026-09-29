import matplotlib.pyplot as plt
from matplotlib import colors
import qutrits
import numpy as np
import cmath
import fns
import Hadamard
import scipy
from scipy import optimize
from scipy.optimize import minimize


phi = 0
t_max = 5e-6
del_t = 1e-8
tau = 8.5e-7
t_s = np.arange(0, t_max, del_t)
exp = cmath.exp(1j*phi)
del_max=0

def del_fn(y):
    return 0
def rabi_fn(y,tau):
    return fns.square_rabi(y,t_max,tau,rabi_max)/(2*np.pi)/1e6



l=2
n=1


shift =10

def dif(rabi_max):
    import cmath

    del_t = 1e-8
    tau = 8.5e-7
    phi = 0
    exp = cmath.exp(1j * phi)
    psi_0 = np.array([0, 1])
    psi_0 = psi_0 / np.sqrt(4)


    P = (0j) * np.zeros((len(t_s), 2))
    P[0] = psi_0
    Prob_t = np.zeros((2, len(t_s)))
    for i in range(2):
        Prob_t[i][0] = abs(P[0][i]) ** 2

    for i in range(len(t_s) - 1):
        H_1 = fns.qubit_H(0, fns.square_rabi(t_s[i],t_max,tau,rabi_max), exp)

        for n in range (2):
            Prob_t[n][i+1] = abs(P[i][n])**2

        P[i + 1] = np.dot(scipy.linalg.expm(-1j * H_1 * del_t), P[i])

    shift = abs(Prob_t[0][0] - Prob_t[-1][0])
    return shift

def prob(rabi_max):
    del_t = 1e-8
    tau = 8.5e-7
    phi = 0
    exp = cmath.exp(1j * phi)
    psi_0 = np.array([0, 1])
    psi_0 = psi_0 / np.sqrt(4)


    P = (0j) * np.zeros((len(t_s), 2))
    P[0] = psi_0
    Prob_t = np.zeros((2, len(t_s)))
    for i in range(2):
        Prob_t[i][0] = abs(P[0][i]) ** 2

    for i in range(len(t_s) - 1):
        H_1 = fns.qubit_H(0, fns.square_rabi(t_s[i],t_max,tau,rabi_max), exp)

        for n in range (2):
            Prob_t[n][i+1] = abs(P[i][n])**2

        P[i + 1] = np.dot(scipy.linalg.expm(-1j * H_1 * del_t), P[i])

    return Prob_t

# psi_0 = [0,1,0,0,1,0,0,0,0]
# psi_1 = [1,0,0,1,0,0,0,0,0]
#
# def Had(design_variables):
#     return np.linalg.norm(Hadamard.Hadamard_20(psi_0, design_variables[0], design_variables[1])[1] - psi_1)
#
#
# rabi_max = 1e6 * 2
# del_max = 7e6
# print(dif(rabi_max))
#
# x0 = [rabi_max, del_max]
# res = scipy.optimize.minimize(Had, x0=x0, method = 'COBYQA')
# # print(dif(tau+1e-7))
#
# fig, axs = plt.subplots(2+2,1)
# labels = [r'$P_{|0\rangle}$',r'$P_{|1\rangle}$']
# for i in range (2):
#     axs[i+1].plot(t_s, prob(rabi_max)[i-1])
#     axs[i+1].set_xlabel(r'$Time$ / $\mu s$')
#     axs[i+1].set_ylabel(labels[i])
#     axs[i+1].set_ylim([-0.1, 1.1])
#
# axs[4].plot()
# plt.show()
#
# print(psi_0)
# # print(psi_1)
# print(res.fun)
# print(res.x)
# print(res.message)

del_max_2 = 0
P = []
plt.figure()
for i in range (100):
    del_max = 1e8
    rabi_max = 4.6490752e6
    tau = 1e-6



    Had = Hadamard.Hadamard_2([1, 0],del_max_2)
    # Had_2 = Hadamard.Hadamard_2(Had_1[1][-1], del_max_2)
    # Had_3 = Hadamard.Hadamard_1(Had_2[1][-1],rabi_max, del_max, tau)
    # Had = Hadamard.Hadamard_2(Had_3[1][-1], del_max_2)

    P.append(Had[1][-1][0].real)
    Prob_t = Had[0]

    del_max_2 += 5e5

plt.plot(np.arange(0,del_max_2,5e5),P)
plt.show()