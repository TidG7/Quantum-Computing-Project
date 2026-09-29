import numpy as np
import cmath
import matplotlib.pyplot as plt
from matplotlib import colors
import fns
import scipy
from scipy import constants
from scipy import optimize
import qutip
import CZ

from variables import del_max

hbar = scipy.constants.hbar
labels_qubit = [r'$P_{|0\rangle}$', r'$P_{|1\rangle}$']

Hadamard = np.matrix(1/np.sqrt(2)*np.array([[1,1],[1,-1]]))

def Hadamard_trit(psi, del_fn, rabi_fn, del_max,rabi_1,rabi_2,tau):
    iden = np.array([[1, 0,0], [0, 1,0],[0,0,0]])

    rabi_3 = -1 / 2 * np.pi * 1e6

    t_max = 3 * tau
    del_t = 1e-8

    t_s = np.arange(0, t_max, del_t)



    P = (0j) * np.zeros((len(t_s), 9))
    P[0] = psi
    Prob_t = np.zeros((9, len(t_s)))
    for i in range(2):
        Prob_t[i][0] = abs(P[0][i]) ** 2

    for i in range(len(t_s) - 1):
        if t_s[i] <= tau:
            P[i + 1] = np.dot(scipy.linalg.expm(-1j * np.kron(np.identity(3),fns.qutrit_H01(del_fn(t_s[i], t_max, del_max),rabi_fn(t_s[i], t_max, tau, rabi_1),1)) * del_t), P[i])
        elif tau < t_s[i] <= 2 * tau:
            P[i + 1] = np.dot(scipy.linalg.expm(-1j * np.kron(np.identity(3),fns.qutrit_H01(del_fn(t_s[i], t_max, del_max),rabi_fn(t_s[i], t_max, tau, rabi_2),1j)) * del_t), P[i])
        elif 2 * tau < t_s[i] <= t_max:
            P[i + 1] = np.dot(scipy.linalg.expm(-1j * rabi_3 * np.kron(np.identity(3),iden) * del_t), P[i])

        for n in range(9):
            Prob_t[n][i + 1] = abs(P[i + 1][n]) ** 2

    return Prob_t, P[-1], t_s


def Hadamard_1(psi, del_fn, rabi_fn, del_max,rabi_1,rabi_2,tau):
    iden = np.array([[1, 0], [0, 1]])
    rabi_3 = -1 / 2 * np.pi * 1e6

    t_max = 3 * tau
    del_t = 1e-8

    t_s = np.arange(0, t_max, del_t)

    P = (0j) * np.zeros((len(t_s), 2))
    P[0] = psi
    Prob_t = np.zeros((2, len(t_s)))
    for i in range(2):
        Prob_t[i][0] = abs(P[0][i]) ** 2

    for i in range(len(t_s) - 1):
        if t_s[i] <= tau:
            P[i + 1] = np.dot(scipy.linalg.expm(-1j * fns.qubit_H(del_fn(t_s[i], tau, del_max),rabi_fn(t_s[i], tau, tau, rabi_1),1) * del_t), P[i])
        elif tau < t_s[i] <= 2 * tau:
            P[i + 1] = np.dot(scipy.linalg.expm(-1j * fns.qubit_H(del_fn(t_s[i], tau, del_max),rabi_fn(t_s[i]-tau, tau, tau, rabi_2),1j) * del_t), P[i])
        elif 2 * tau < t_s[i] <= t_max:
            P[i + 1] = np.dot(scipy.linalg.expm(-1j * rabi_3 * iden * del_t), P[i])

        for n in range(2):
            Prob_t[n][i + 1] = abs(P[i + 1][n]) ** 2

    return Prob_t, P, t_s

def Hadamard_rho(psi, del_fn, rabi_fn, del_max,rabi_1,rabi_2,tau):
    pauli_x = fns.qubit_H(0, 1, cmath.exp(1j*0))
    pauli_y = fns.qubit_H(0, 1, cmath.exp(1j*np.pi/2))
    iden = np.identity(2)

    rabi_3 = -1 / 2 * np.pi * 1e6

    t_max = 3 * tau
    del_t = 1e-8

    t_s = np.arange(0, tau, del_t)


    def U(rabi, pauli, t):
        return scipy.linalg.expm(1j * rabi * pauli * t)

    rho = (0j) * np.zeros((4, 2, 2))
    rho[0] = np.matrix(psi).getH() @ np.matrix(psi)

    rho[1] = np.matrix(U(rabi_1, pauli_x, tau)) @ rho[0] @ np.matrix(U(rabi_1, pauli_x, tau)).getH()
    rho[2] = np.matrix(U(rabi_2, pauli_y, tau)) @ rho[1] @ np.matrix(U(rabi_2, pauli_y, tau)).getH()
    rho[3] = np.matrix(U(rabi_3, iden, tau)) @ rho[2] @ np.matrix(U(rabi_3, iden, tau)).getH()

    result = qutip.mesolve(qutip.Qobj(pauli_x * rabi_1), qutip.Qobj(psi), t_s, c_ops=0*qutip.Qobj([[0,1],[0,0]]))


    return rho, result

def Hadamard_col(psi, del_fn, rabi_fn, del_max,rabi_1,rabi_2,tau, c_ops):
    pauli_x = fns.qubit_H(0, 1, cmath.exp(1j*0))
    pauli_y = fns.qubit_H(0, 1, cmath.exp(1j*np.pi/2))
    iden = np.identity(2)

    rabi_3 = -1 / 2 * np.pi * 1e6

    t_max = 3 * tau
    del_t = 1e-8

    t_s = np.arange(0, tau, del_t)

    rhos = []

    rhos_1 = qutip.mesolve(qutip.Qobj(pauli_x * rabi_1), qutip.Qobj(psi), t_s, c_ops=qutip.Qobj(c_ops)).states

    rhos.append(rhos_1)
    rhos_2 = qutip.mesolve(qutip.Qobj(pauli_y * rabi_2), rhos_1[-1], t_s, c_ops=qutip.Qobj(c_ops)).states

    rhos.append(rhos_2)
    rhos_3 = qutip.mesolve(qutip.Qobj(iden* rabi_3), rhos_2[-1], t_s, c_ops=qutip.Qobj(c_ops)).states

    rhos.append(rhos_3)

    return rhos

def Hadamard_trit_col(psi, del_fn, rabi_fn, del_max,rabi_1,rabi_2,tau, c_ops):
    pauli_x = fns.qutrit_H01(0, 1, cmath.exp(1j*0))
    pauli_y = fns.qutrit_H01(0, 1, cmath.exp(1j*np.pi/2))
    iden = fns.qutrit_H01(1, 0, cmath.exp(1j*0))

    rabi_3 = 3 / 2 * np.pi * 1e6

    del_t = 1e-8

    t_s = np.arange(0, tau, del_t)

    rhos = []

    rhos_1 = qutip.mesolve(qutip.Qobj(pauli_x * rabi_1), qutip.Qobj(psi), t_s, c_ops=c_ops).states

    rhos.append(rhos_1)
    rhos_2 = qutip.mesolve(qutip.Qobj(pauli_y * rabi_2), rhos_1[-1], t_s, c_ops=c_ops).states

    rhos.append(rhos_2)
    rhos_3 = qutip.mesolve(qutip.Qobj(iden* rabi_3), rhos_2[-1], t_s, c_ops=c_ops).states

    rhos.append(rhos_3)

    return rhos


def Hadamard_trit_qut(psi, del_fn, rabi_fn, del_max, rabi_1, rabi_2, tau, c_ops):
    pauli_x = fns.qutrit_H01(0, 1, cmath.exp(1j * 0))
    pauli_y = fns.qutrit_H01(0, 1, cmath.exp(1j * np.pi / 2))
    iden = fns.qutrit_H01(1, 0, cmath.exp(1j * 0))

    rabi_3 = 3 / 2 * np.pi * 1e6

    del_t = 1e-8

    t_s = np.arange(0, tau, del_t)

    rhos = []

    rhos_1 = qutip.mesolve(qutip.Qobj(pauli_x * rabi_1), qutip.Qobj(psi), t_s).states

    rhos.append(rhos_1)
    rhos_2 = qutip.mesolve(qutip.Qobj(pauli_y * rabi_2), rhos_1[-1], t_s).states

    rhos.append(rhos_2)
    rhos_3 = qutip.mesolve(qutip.Qobj(iden * rabi_3), rhos_2[-1], t_s).states

    rhos.append(rhos_3)

    return rhos

def Hadamard_qut(psi, del_fn, rabi_fn, del_max,rabi_1,rabi_2,tau):
    pauli_x = fns.qubit_H(0, 1, cmath.exp(1j*0))
    pauli_y = fns.qubit_H(0, 1, cmath.exp(1j*np.pi/2))
    iden = np.identity(2)

    rabi_3 = -1 / 2 * np.pi * 1e6

    t_max = 3 * tau
    del_t = 1e-8

    t_s = np.arange(0, tau, del_t)

    rhos = []

    rhos_1 = qutip.mesolve(qutip.Qobj(pauli_x * rabi_1), qutip.Qobj(psi), t_s).states

    rhos.append(rhos_1)
    rhos_2 = qutip.mesolve(qutip.Qobj(pauli_y * rabi_2), rhos_1[-1], t_s).states

    rhos.append(rhos_2)
    rhos_3 = qutip.mesolve(qutip.Qobj(iden * rabi_3), rhos_2[-1], t_s).states

    rhos.append(rhos_3)

    return rhos


def contour(prob, t_s, labels, axs):

    X, Y = np.meshgrid(t_s, [0, 1])

    for i in range(len(prob[0])):
        norm = colors.Normalize(vmin=np.min(prob), vmax=np.max(prob))

        Z = [prob[i], prob[i]]

        cm = axs[i + 1].contourf(X, Y, Z, 10, extend="max", norm=norm)
        axs[i + 1].set_ylabel(labels[i])
        axs[i + 1].set_ylim([0, 1])

def param(del_fn, del_max, rabi_fn, rabi_1, taus):
    det = [del_fn(i,taus[-1],del_max)/(2*np.pi) for i in taus]
    rab = [rabi_fn(i,taus[-1],tau,rabi_1)/(2*np.pi) for i in taus]

    plt.plot(taus*1e6, rab, label = '$\Omega/2\pi$')
    plt.plot(taus*1e6,det, label = '$\Delta/2\pi$')
    # plt.legend()


if __name__ == '__main__':
    psi_0=[1,0]
    psi_0=psi_0/scipy.linalg.norm(psi_0)
    del_fn = fns.single_del
    rabi_fn = fns.square_rabi
    del_max = 0
    rabi_1 = np.pi/2 * 1e6
    rabi_2 = np.pi/4 *1e6

    gamma=4e2
    c_ops = np.array([[0, 1], [0, 0]])

    tau = 1e-6
    del_t = 1e-8

    prob, psi, t_s = Hadamard_1(psi_0,del_fn,rabi_fn,del_max,rabi_1,rabi_2,tau)


    fig, axs = plt.subplots(2, 3)
    rhos_col = Hadamard_col(psi_0, del_fn, rabi_fn, del_max, rabi_1, rabi_2, tau,0*c_ops)
    print(rhos_col[-1][-1])

    qutip.hinton(rhos_col[0][-1], ax=axs[1][0])
    qutip.hinton(rhos_col[1][-1], ax=axs[1][1])
    qutip.hinton(rhos_col[2][-1], ax=axs[1][2])

    rhos_qut = Hadamard_qut(psi_0, del_fn, rabi_fn, del_max, rabi_1, rabi_2, tau)

    qutip.hinton(rhos_qut[0][-1] * rhos_qut[0][-1].dag(), ax=axs[0][0])
    qutip.hinton(rhos_qut[1][-1] * rhos_qut[1][-1].dag(), ax=axs[0][1])
    qutip.hinton(rhos_qut[2][-1] * rhos_qut[2][-1].dag(), ax=axs[0][2])

    plt.show()

    rabi_1s = np.arange(0e6,4e6,1e4)
    psis0 = []
    psis1 = []

    taus = np.arange(0,tau,del_t)
    fig, axs = plt.subplots(3, 1)
    for j in range(len(rabi_1s)):
        P=0j*np.zeros((len(taus),2))
        P[0]=psi[0]
        prob=np.zeros((2,len(taus)))
        for i in range(2):
            prob[i][0] = abs(P[0][i]) ** 2
        for i in range(len(taus)-1):
            P[i + 1] = np.dot(scipy.linalg.expm(-1j * fns.qubit_H(del_fn(taus[i], tau, del_max), rabi_fn(taus[i], tau, tau, rabi_1s[j]),1j) * del_t), P[i])
            for n in range(2):
                prob[n][i + 1] = abs(P[i + 1][n]) ** 2
        psis0.append(P[-1][0].real)
        psis1.append(P[-1][1].real)


        det = [del_fn(i, tau, del_max) for i in taus]
        rab = [rabi_fn(i, tau, tau, rabi_1s[j]) for i in taus]


        axs[0].plot(taus,det)
        axs[0].plot(taus,rab)

        axs[1].plot(taus,prob[0])
        axs[2].plot(taus,prob[1])



    plt.figure()
    plt.plot(rabi_1s,psis0,label = '0')
    plt.plot(rabi_1s,psis1, label='1')
    plt.legend()



    def func(x, a, b, c):
        return a * np.sin(b * (x+c) )




    *params1, params_covariance1 = optimize.curve_fit(func, rabi_1s, psis0, p0=[.5, 1/1.385e7*2*np.pi,-2.3e6])
    *params2, params_covariance2 = optimize.curve_fit(func, rabi_1s, psis1, p0=[.5, 1 / 1.385e7 * 2 * np.pi, -2.3e6])

    print(params1)
    print(params2)
    plt.plot(rabi_1s, [params1[0][0]*np.sin((i+params1[0][2])*params1[0][1]) for i in rabi_1s])
    plt.plot(rabi_1s, [params2[0][0]*np.sin((i+params2[0][2])*params2[0][1]) for i in rabi_1s])

    rho_Had = Hadamard @ (np.matrix(psi_0).getH() @ np.matrix(psi_0)) @ Hadamard.getH()
    rho_2 = np.matrix(psi[-1]).getH() @ np.matrix(psi[-1])
    rho_3 = Hadamard_rho(psi_0, del_fn,rabi_fn, del_max, rabi_1, rabi_2, tau)[0][-1]
    print(qutip.fidelity(qutip.Qobj(rho_Had),qutip.Qobj(rhos_col[-1][-1])))
    print(qutip.fidelity(qutip.Qobj(rho_Had),qutip.Qobj(rho_2)))
    plt.figure()

    param(del_fn, del_max, rabi_fn, rabi_1, taus)
    plt.xlabel('t/$\mu$s')
    plt.ylabel('MHz')
    plt.tight_layout()
    # qutip.visualization.matrix_histogram_complex(qutip.Qobj(rho_2))
    plt.figure()

    psi_0s = [np.array([1,0]),np.array([0,1]),np.array([1,1])/np.sqrt(2)]
    gam_rs = np.arange(0,1.5e3,20)
    fid_ave = []
    fids = []
    for i in gam_rs:
        fid=[]
        for j in psi_0s:
            rho_1 = Hadamard_col(j, del_fn, rabi_fn, del_max, rabi_1, rabi_2, 1e-6, c_ops*i)[-1][-1]
            rho_Had = Hadamard @ (np.matrix(j).getH() @ np.matrix(j)) @ Hadamard.getH()
            fid.append(qutip.fidelity(qutip.Qobj(rho_Had),qutip.Qobj(rho_1)))

        fid_ave.append(np.average(fid))
        fids.append(fid)


    def func_2(x, a, b, c):
        return a * np.exp(-b * (x)) + c




    # *params2, params_covariance2 = optimize.curve_fit(func_2, gam_rs, fid_ave, p0=[1, 1 / 2000, 0.7])
    # print(params2)
    # print(params2[0])
    # model = [func_2(i, params2[0][0], params2[0][1], params2[0][2]) for i in gam_rs]
    # print(model)
    # st_dev = np.array([model, fid_ave]).std(axis=0)

    plt.scatter((gam_rs**2)/1e6,fid_ave, marker='.')
    # plt.scatter((gam_rs**2)/1e6,CZ.fid_plot(gam_rs), marker = '.')
    # plt.plot(gam_rs**2,fids)
    # plt.plot(gam_rs**2,model)

    plt.xlabel('$\gamma/MHz$')
    plt.ylabel('fidelity')
    print(fid_ave[0])
    #         Had_1 = Hadamard_1([1, 0],rabi_1, del_1, tau)
    # Had_2 = Hadamard_2(Had_1[1][-1],rabi_2, del_2, tau)
    # Had = Hadamard_1(Had_2[1][-1],rabi_3, del_3, tau)
    # Had = Hadamard_1([1,0],rabi_4, del_4, tau)




    # fig, axs = plt.subplots(2+3,1)
    # linewidth = 1
    # color = 'b'
    # labels = [r'$P_{|00\rangle}$',r'$P_{|01\rangle}$', r'$P_{|0r\rangle}$', r'$P_{|10\rangle}$', r'$P_{|11\rangle}$',r'$P_{|1r\rangle}$', r'$P_{|r0\rangle}$', r'$P_{|r1\rangle}$', r'$P_{|rr\rangle}$']
    # for i in range (2):
    #     axs[i+1].plot(t_s, Prob_t[i], linewidth = linewidth, color = color)
    #     axs[i+1].set_xlabel(r'$Time$ / $\mu s$')
    #     axs[i+1].set_ylabel(labels[i])
    #     axs[i+1].set_ylim([-0.1, 1.1])
    #
    # axs[3].plot(t_s,psi_t[:,0].imag, label = 'Imaginary')
    # axs[3].plot(t_s, psi_t[:, 0].real, label = 'Real')
    # axs[3].plot(t_s, abs(psi_t[:, 1]), label = 'abs')
    # axs[3].legend()
    #
    # axs[4].plot(t_s, psi_t[:, 1].imag, label='Imaginary')
    # axs[4].plot(t_s, psi_t[:, 1].real, label='Real')
    # axs[4].plot(t_s, abs(psi_t[:, 0]), label='abs')
    # axs[4].legend()
    #
    # def del_fn(y):
    #     return fns.sine_del(y, t_max, del_max)
    #
    # rabi_ = [fns.adiabatic_rabi(i,t_max,tau, rabi_max)/(2*np.pi)/1e6 for i in t_s]
    # del_ = [del_fn(i)/(2*np.pi)/1e6 for i in t_s]
    #
    # twin = axs[0].twinx()
    # axs[0].plot(t_s, rabi_, linewidth=1, color='r', label = '$\Omega/2\pi$')
    # twin.plot(t_s, del_, linewidth=1, color='b', label = '$\Delta/2\pi$')
    # axs[0].set_xlabel(r'$Time$ / $\mu s$')
    # axs[0].set_ylabel(r'MHz')
    # plt.legend()
    #
    # print(Prob_t[:,-1])
    # print(psi_t[-1])
    # print(Hadamard_20([0,1,0,0,1,0,0,0,0], del_max, rabi_max)[1][-1])

    plt.show()
    print(np.dot(scipy.linalg.expm(-1j * fns.qubit_H(0,np.pi/4*1e6, 1j) * tau), [1,0]))
