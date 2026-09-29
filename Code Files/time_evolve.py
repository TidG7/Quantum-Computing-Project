import numpy as np
import eigen
import cmath


def psi(psi_0, H, t):
    psi_t = np.array(np.matmul(evolve(H, t), psi_0))
    M = []

    for i in range(len(psi_t)):
        M.append(psi_t[i])
    return M

# getting diagonal and rotation matrices
def transform(H, t):
    psi = np.asmatrix(eigen.eigenvectors(H))



    R = np.asmatrix(psi)
    R = np.transpose(R)

    D = (1j) * np.zeros((len(H), len(H)))

    for i in range(len(D[0])):
        D[i][i] = eigen.q(H, i, t)

    D = np.asmatrix(D)

    return D, R


def evolve(H, t):

    D, R = transform(H, t)

    Q = np.matmul(np.conjugate(R).T, D)

    Q = np.matmul(Q, R)

    return Q
