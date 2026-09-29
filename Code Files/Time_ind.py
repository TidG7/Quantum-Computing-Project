import numpy as np
import cmath
import fns
import time_evolve
import matplotlib.pyplot as plt
import scipy
import statistics
hbar = scipy.constants.hbar

detun = 0
Rabi = 1*10**6 *2*np.pi
phi = 0
exp = cmath.exp(1j*phi)

psi_0 = [1,0,0,0]
t_s = np.arange(0,2, 1e-2)
Q = np.identity(2)

def analytic_0(t, rabi):
    return (np.cos(t*rabi/2)**4)
def analytic_1(t,f):
    return np.cos(t*f/2)**2

analytic_0 = np.array([analytic_0(i*1e-6, Rabi) for i in t_s])



V = 0
linewidth = 1
color = 'blue'
fig, axs = plt.subplots(3)

for i in range (100):

    P_00 = np.zeros(len(t_s))
    P_phi = np.zeros(len(t_s))

    H_V = np.zeros([4,4])
    H_V[-1,-1] = 2*V

    H_1 = fns.H(detun, Rabi, exp)
    H_2 = hbar/2 * (np.kron(Q, H_1) + np.kron(H_1, Q) + H_V)

    for i in range(len(t_s)):
        P = time_evolve.psi(psi_0, H_2, t_s[i]*1e-6)[0]
        P_00[i] = (abs(P[0]) ** 2)
        P_phi[i] = 1/np.sqrt(2) * (abs(P[1]) ** 2 + abs(P[2]) ** 2)


    axs[0].plot(t_s, P_00, linewidth = linewidth, color = color)
    axs[0].set_ylabel(r'$P_{|00\rangle}$')
    axs[0].set_ylim([0,1])


    axs[2].plot(t_s, P_phi, linewidth = linewidth, color = color)
    axs[2].set_ylabel(r'$P_{|\psi^+\rangle}$')
    axs[2].set_ylim([0,1])

    if V == 0:
        res = P_00 - analytic_0
    V += (1 * 10**6)
    linewidth = 0.05
    color = 'red'

# axs[0][1].set_xlim([t_s[0],t_s[-1]])
f=8.879081215997914
analytic_2 = np.array([analytic_1(i, f) for i in t_s])
dev = []
dev = dev + [statistics.stdev(P_00-analytic_2)]
while True:
    f += 1e-12
    analytic_2 = np.array([analytic_1(i, f) for i in t_s])
    res_1 = np.array(P_00) - analytic_2
    dev = dev + [statistics.stdev(res_1)]
    if dev[-1] > dev[-2]:
        break

test = [analytic_1(i,8.5) for i in t_s]
axs[0].plot(t_s,analytic_2)
# axs[0].plot(t_s,test)
print(f)

res_2 = P_00-analytic_2

axs[1].scatter(t_s[0::19], res[0::19]*1e15, marker='^', color = 'g', linewidths=.1)
axs[1].scatter(t_s[0::19], res_2[0::19]*1e4, marker='*', color = 'orange', linewidths=.1)

# axs[0].plot(t_s, analytic_0, color = 'g', linewidth = 0.5)
# axs[0].plot(t_s, analytic_1, color = 'b', linewidth = 0.1)

axs[2].set_position([.1,.14,.8,.3])
axs[1].set_position([.1,.55,.8,.09])
axs[0].set_position([.1,.64,.8,.3])

# axs[1].set_ylim([-15,15])
# axs[1].set_yticks([-8,0,8])

axs[1].set_ylabel(r'e / $10^{-15}$')
axs[1].set_xlabel(r'$Time$ / $\mu s$')
axs[2].set_xlabel(r'$Time$ / $\mu s$')

plt.show()
fig.savefig('Time-ind-H.png')

