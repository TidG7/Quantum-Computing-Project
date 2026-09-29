import numpy as np
import cmath
import matplotlib.pyplot as plt
import fns
import scipy
from scipy import constants
hbar = scipy.constants.hbar

phi = 0
exp = cmath.exp(1j*phi)
t_max = 5e-6
del_t = 1e-8
tau = 8e-7
t_s = np.arange(0, t_max, del_t)

del_max = 1e7
rabi_max = 2.887e6

psi_0 = np.array([0,1])

psi_0 = psi_0/np.sqrt(1)
print(psi_0)

fig, axs = plt.subplots(2+1,1)
print(psi_0)
linewidth = 1
color = 'b'

P = (0j) * np.zeros((len(t_s), 2))
P[0] = psi_0

rho = (0j) * np.zeros((len(t_s), 2,2))
rho[0] = np.matrix(P[0]).getH() @ np.matrix(P[0])

Prob_t = np.zeros((2, len(t_s)))
for i in range (2):
    Prob_t[i][0] = abs(P[0][i])**2

for i in range(len(t_s)-1):
    H_1 = fns.qubit_H(fns.sine_del(t_s[i],t_max,del_max), fns.square_rabi(t_s[i],t_max,tau,rabi_max), exp)

    rho[i+1] = np.matrix(P[i]).getH() @ np.matrix(P[i])

    for n in range (2):
        Prob_t[n][i+1] = abs(P[i+1][n])**2

    P[i+1] = np.dot(scipy.linalg.expm(-1j * H_1 * del_t), P[i])

plt.figure(figsize=(2,2))
plt.imshow(abs(rho[1]))
plt.show()

labels = [r'$P_{|0\rangle}$',r'$P_{|1\rangle}$']
for i in range (2):
    axs[i+1].plot(t_s, Prob_t[i], linewidth = linewidth, color = color)
    axs[i+1].set_xlabel(r'$Time$ / $\mu s$')
    axs[i+1].set_ylabel(labels[i])
    axs[i+1].set_ylim([-0.1, 1.1])

print(P[200])
print(P[-1])

rabi_ = [fns.square_rabi(i,t_max,tau,rabi_max)/(2*np.pi)/1e6 for i in t_s]
del_ = [fns.sine_del(i, t_max, del_max)/(2*np.pi)/1e6 for i in t_s]

twin = axs[0].twinx()
axs[0].plot(t_s, rabi_, linewidth=1, color='r', label = '$\Omega/2\pi$')
twin.plot(t_s, del_, linewidth=1, color='b', label = '$\Delta/2\pi$')
axs[0].set_xlabel(r'$Time$ / $\mu s$')
axs[0].set_ylabel(r'MHz')
# plt.legend(axs[0],twin, loc=0)

print(Prob_t[:,-1])

fig.tight_layout()
plt.show()