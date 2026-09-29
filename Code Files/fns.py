import numpy as np
def linear_del(t, t_max, del_max):
    return ((100 / t_max) * t - 50) * del_max

def cubic_del(t, t_max, del_max):
    return 8 * ((t-t_max/2)/t_max)**3 * del_max

def single_del(t, t_max, del_max):
    return del_max

def double_square_rabi(t, t_max, tau, rabi_max):
    if 2.5*t_max/12-tau/2 <= t <= 2.5*t_max/12+tau/2 or 9.5*t_max/12-tau/2 <= t <= 9.5*t_max/12+tau/2:
        return rabi_max
    elif t_max/2-tau/4 <= t <= t_max/2+tau/4:
        return rabi_max/2
    else:
        return 0

def square_rabi(t, t_max, tau, rabi_max):
    return rabi_max

def adiabatic_rabi(t, t_max, tau, rabi_max):
    return rabi_max*(np.exp((-(2*t-t_max)**4)/(tau/2)**4) + 2*t*(2*t-t_max))
def gauss_rabi(t, t_max, tau, rabi_max):
    return rabi_max*np.exp(-100*(t-tau/2)**2/(4*(tau**2)))


def sine_del(t, t_max, del_max):
    return -del_max*np.cos(np.pi * t/(t_max))


# del_max*np.cos(np.pi * (t-t_max/2+t_max/12)/(t_max/6))

def qubit_H(delta, rabi, exp):
    return np.array(([delta, rabi*exp], [rabi/exp, -delta]))

def qutrit_H01(delta, rabi, exp):
    return np.array(([delta,rabi*exp,0], [rabi/exp,-delta,0], [0,0,0]))
def qutrit_H12(delta, rabi, exp):
    return np.array(([0,0,0], [0,delta,rabi*exp], [0,rabi/exp,-delta]))
def qutrit_H02(delta, rabi, exp):
    return np.array(([delta,0,rabi*exp], [0,0,0], [rabi/exp,0,-delta]))

import matplotlib.pyplot as plt

# t_max = 5e-6
# del_t = 1e-8
# tau = 6e-7
# t_s = np.arange(0, t_max, del_t)
# rabi_max = 8e6
#
# plt.figure()
# rabi_ = [adiabatic_rabi(i,t_max,tau,rabi_max) for i in t_s]
#
# plt.plot(t_s, rabi_, linewidth=1, color='r')
# plt.show()