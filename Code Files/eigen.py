from numpy import linalg as LA
import cmath
import scipy
import numpy as np

hbar = scipy.constants.hbar

def eigenvalues(A):
    return LA.eig(A)[0]
def eigenvectors(A):
    return LA.eig(A)[1]

def q(H, num, t):
    return cmath.exp(-1j * eigenvalues(H)[num] * t/hbar)
