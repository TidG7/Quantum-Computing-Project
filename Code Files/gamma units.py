import numpy as np
import cmath
import matplotlib.pyplot as plt
from matplotlib import colors
import fns
import scipy
from scipy import constants
from scipy import optimize
import qutip
import Hadamard


psi_0=[0,1]
psi_0=psi_0/scipy.linalg.norm(psi_0)
del_fn = fns.single_del
rabi_fn = fns.gauss_rabi
del_max = 0
rabi_1 = 0
rabi_2 = 0

gams=np.arange(0,1e12,1e5)
c_ops = np.array([[0, 1], [0, 0]])

tau = 1e-6
del_t = 1e-8

rho = Hadamard.Hadamard_col(psi_0,del_fn,rabi_fn,del_max,rabi_1,rabi_2,tau,c_ops)
print(rho)
