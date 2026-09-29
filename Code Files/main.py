import matplotlib.pyplot as plt
from matplotlib import colors
import qutrits
import numpy as np
import cmath
import fns
import Hadamard
import scipy

phi = 0
t_max = 8e-7 * 4
del_t = 1e-8
tau = 8.5e-7*4
t_s = np.arange(0, t_max, del_t)
exp = cmath.exp(1j*phi)

del_max = 1e5
rabi_max = 2e6

V=1e10

def del_fn(y):
    return fns.sine_del(y,t_max,del_max)
def rabi_fn(y,tau):
    return fns.square_rabi(y,t_max,tau,rabi_max)/(2*np.pi)/1e6

psi_0 = np.array([1,0,0,1,0,0,0,0,0])
psi_0 = psi_0/scipy.linalg.norm(psi_0)

l=3
n=2

# prob_10 = Hadamard.Hadamard([psi_0[0], psi_0[1]])[0]
# prob_11 = Hadamard.Hadamard([psi_0[4], psi_0[5]])[0]
# psi_10 = Hadamard.Hadamard([psi_0[0], psi_0[1]])[1][-1]
# psi_11 = Hadamard.Hadamard([psi_0[4], psi_0[5]])[1][-1]

# psi_1 = np.concatenate((psi_10, psi_0[2], psi_11, psi_0[5:9]))
prob_1, psi_1, t_s1 = Hadamard.Hadamard_trit(psi_0)
labels, prob_2, psi_2 = qutrits.two_qutrit(psi_1, del_max, rabi_max, V, t_max, del_t, tau, exp)
prob_3, psi_3, t_s3 = Hadamard.Hadamard_trit(psi_2)

# prob_20 = Hadamard.Hadamard([psi_2[0], psi_2[1]])[0]
# prob_21 = Hadamard.Hadamard([psi_2[4], psi_2[5]])[0]
# psi_20 = Hadamard.Hadamard([psi_2[0], psi_2[1]])[1][-1]
# psi_21 = Hadamard.Hadamard([psi_2[4], psi_2[5]])[1][-1]

# psi_3 = np.concatenate((psi_20, psi_2[2], psi_21, psi_2[5:9]))


prob = np.concatenate((prob_1, prob_2, prob_3), axis=1)
fig, axs = plt.subplots(l**n+1,1)

t_2 = np.array(t_s) + t_s1[-1]
t_3 = np.array(t_s3) + t_2[-1]
t=np.concatenate((t_s1 ,t_2, t_3))

X, Y = np.meshgrid(t, [0, 1])
for i in range(l ** n):
    norm = colors.Normalize(vmin=np.min(prob), vmax=np.max(prob))

    Z = [prob[i], prob[i]]

    cm = axs[i + 1].contourf(X, Y, Z, 40, extend="max", norm=norm)
    axs[i + 1].set_ylabel(labels[i])
    axs[i + 1].set_ylim([0, 1])
    if i == 3:
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.65])
        fig.colorbar(cm, cax=cbar_ax)
print(psi_3)

rho = np.matrix(psi_3).getH() @ np.matrix(psi_3)

# plt.figure(figsize=(2,2))
# plt.imshow(abs(rho))


axs[-1].set_xlabel(r'$Time$ / $\mu s$')



rabi_=[]
del_=[]
for i in t:
    if i <= t_max:
        rabi_.append(rabi_fn(i, tau)/(2*np.pi))
        del_.append(del_fn(i)/(2*np.pi))
    elif i <= 2*t_max:
        rabi_.append(rabi_fn(i-t_max, tau)/(2*np.pi))
        del_.append(del_fn(i-t_max)/(2*np.pi))
    else:
        rabi_.append(rabi_fn(i - 2*t_max, tau)/(2*np.pi))
        del_.append(del_fn(i - 2*t_max)/(2*np.pi))

twin = axs[0].twinx()
axs[0].plot(t, rabi_, color='r', label = '$\Omega$')
axs[0].legend()
twin.plot(t, del_, color='b', label = '$\Delta$')
axs[0].set_ylabel(r'$\Omega/2\pi$ MHz')
twin.set_ylabel(r'$\Delta/2\pi$ MHz')
twin.legend(loc=1)


plt.show()