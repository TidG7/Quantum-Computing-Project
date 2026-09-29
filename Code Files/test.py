import numpy as np
import cmath
import scipy
import matplotlib.pyplot as plt
import qutip
from qutip import *
#
# # set variables
# t_max = 1e-6
# psi_0 = np.array([1,0,0,0])
# del_t = 5e-9
# t_s = np.arange(0,t_max, del_t)
# Rabi = 1e6
# phi = np.pi
#
#
# psi_0 = psi_0 * 1 /(sum(psi_0**2)**(1/2))
# exp = cmath.exp(1j*phi)
#
#
# # the time dependent delta function
# def detun(t):
#     return ((100/t_max) * t - 50)*1e6
# def psi(psi_0, H, t):
#     psi_t = np.array(np.matmul(Time_evolve.evolve(H, t), psi_0))
#
#     M = []
#     for i in range(len(psi_t)):
#         M.append(psi_t[i])
#
#     return M
# def H(t, V, ):
#     H = hbar / 2 * np.array([[2 * detun(t), Rabi / exp, Rabi / exp, 0],
#                                  [Rabi * exp, 0, 0, Rabi / exp],
#                                  [Rabi * exp, 0, 0, Rabi / exp],
#                                  [0, Rabi * exp, Rabi * exp, -2 * detun(t) + 2 * V]])
#
#     return H

# P = (1 + 0j) * np.zeros((len(t_s), 4))
# P[0] = psi_0
#
# P_00 = [abs(psi_0[0])**2]
# P_psi = [abs(psi_0[1])**2 + abs(psi_0[2])**2]
# V=1
# for i in range (len(t_s)-1):
#     H = hbar / 2 * np.array([[2 * detun(t_s[i]), Rabi / exp, Rabi / exp, 0],
#                                         [Rabi * exp, 0, 0, Rabi / exp],
#                                         [Rabi * exp, 0, 0, Rabi / exp],
#                                         [0, Rabi * exp, Rabi * exp, -2 * detun(t_s[i]) + 2 * V]])
#     H_2 = np.matmul(H,H)
#     p = np.array(np.array(P[i]) - 1j * del_t/hbar * np.array(H.dot(P[i])) - (del_t/hbar)**2 * np.array(H_2.dot(np.array(P[i]))))
#     P[i+1] = p
# print(P)
#
# for i in range (100):
#     P= []
#     P.append(psi_0)
#
#     P_00 = [abs(psi_0[0])**2]
#     P_psi = [abs(psi_0[1])**2 + abs(psi_0[2])**2]
#
#     for i in range(len(t_s)-1):
#
#         H = hbar / 2 * np.asmatrix([[2 * detun(t_s[i]), Rabi / exp, Rabi / exp, 0],
#                                         [Rabi * exp, 0, 0, Rabi / exp],
#                                         [Rabi * exp, 0, 0, Rabi / exp],
#                                         [0, Rabi * exp, Rabi * exp, -2 * detun(t_s[i]) + 2 * V]])
#         H_2 = np.matmul(H,H)
#         p = P[-1] - 1j * del_t/hbar * H.dot(P[-1]) - (del_t/hbar)**2 * H_2.dot(P[-1])
#         P.append(p)
#
#
#         P_00.append(abs(P[-1][0]))
#         P_psi.append(abs(P[-1][2])**2+abs(P[-1][1])**2)

# V=0
# P = (0j) * np.zeros((len(t_s), 4))
# P[0] = psi_0
# fig, axs = plt.subplots(2)
# P = (0j) * np.zeros((len(t_s), 4))
# P[0] = psi_0
#
# P_00 = [abs(psi_0[0])**2]
#
# for i in range(len(t_s) - 1):
#
#         p = P[i+1] - 1j * del_t / hbar * psi(P[i+1], H(t_s)[i], del_t)[0]
#
#         P[i+1] = p
#
#
#         P_00.append(abs(P[i+1][0])**2)
#
#
#
# axs[1].plot(t_s, P_00, linewidth=1)
# print(P)
#P_psi.append(1/np.sqrt(2) * (abs(P[i+1][2])**2+abs(P[i+1][1])**2))
# t= np.arange(0,10,0.05)
# y = [(np.cos(i)**2*cmath.exp(-3j*i)) for i in t]
# plt.figure()
# plt.plot(t, y)
# plt.show()
# fig = plt.figure()
b = qutip.Bloch()
x = (qutip.basis(2, 0) + (1+0j)*qutip.basis(2, 1)).unit()
y = (qutip.basis(2, 0) + (0+1j)*qutip.basis(2, 1)).unit()
z = (qutip.basis(2, 0) + (0+0j)*qutip.basis(2, 1)).unit()

print([x, y, z])
