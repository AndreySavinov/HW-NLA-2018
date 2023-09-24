# encoding: utf-8
# pset2.py

import numpy as np
from scipy.sparse import csr_matrix, diags
# don't forget import packages, e.g. scipy
# but make sure you didn't put unnecessary stuff in here

# INPUT : diag_broadcast - list of diagonals value to broadcast,length equal to 3 or 5; n - integer, band matrix shape.
# OUTPUT : L - 2D np.ndarray, L.shape[0] depends on bandwidth, L.shape[1] = n-1, do not store main diagonal, where all ones;                  add zeros to the right side of rows to handle with changing length of diagonals.
#          U - 2D np.ndarray, U.shape[0] = n, U.shape[1] depends on bandwidth;
#              add zeros to the bottom of columns to handle with changing length of diagonals.
def band_lu(diag_broadcast, n): # 5 pts
    # enter your code here
    if len(diag_broadcast)==3:
        U = np.zeros((n, 2))
        L = np.zeros((1, n-1))
        U[0, 0] = diag_broadcast[1]
        U[:n-1, 1] = diag_broadcast[2]
        for k in range (n-1):
            L[0, k] = diag_broadcast[0]/U[k, 0]
            U[k+1, 0] = diag_broadcast[1] - L[0, k]*diag_broadcast[2]
    if len(diag_broadcast)==5:
        U = np.zeros((n, 3))
        L = np.zeros((2, n-1))
        U[0, 0] = diag_broadcast[2]
        U[0, 1] = diag_broadcast[3]
        U[:n-2, 2] = diag_broadcast[4]
        L[0, 0] = diag_broadcast[1]/U[0, 0]
        U[1, 0] = diag_broadcast[2]-L[0, 0]*U[0, 1]
        for k in range(n-2):
            L[1,k] = diag_broadcast[0]/U[k, 0]
            L[0, k+1] = (diag_broadcast[1]- L[1,k]*U[k, 1])/U[k+1,0]
            U[k+1, 1] = diag_broadcast[3]- U[k,2]*L[0,k]
            U[k+2, 0] = diag_broadcast[2]- L[1,k]*U[k, 2]-U[k+1, 1]*L[0, k+1]
    return L, U


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def gram_schmidt_qr(A): # 5 pts
    # your code is here
    m, n = A.shape
    if m>=n :
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
    else :
        Q = np.zeros((m, m))
        R = np.zeros((m, n))
    k = min(m, n)
    Q[:,0] = A[:, 0]/np.linalg.norm(A[:, 0], 2)
    R[0, 0] = np.dot(A[:, 0].transpose(), Q[:, 0])
    for i in range(1, k):
        Q[:, i] = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(A[:, i].transpose(), Q[:, j])
            Q[:, i] = Q[:, i] - R[j,i]*Q[:, j]
        Q[:, i] = Q[:, i]/np.linalg.norm(Q[:, i], 2)
        R[i, i] = np.dot(A[:, i].transpose(), Q[:, i])
    if m<n :
        for i in range (k, n):
            for j in range(m):
                R[j, i] = np.dot(A[:, i].transpose(), Q[:, j])
    return Q, R
    return Q, R

# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def modified_gram_schmidt_qr(A): # 5 pts
    # your code is here
    m, n = A.shape
    if m>=n :
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
    else :
        Q = np.zeros((m, m))
        R = np.zeros((m, n))
    k = min(m, n)
    Q[:,0] = A[:, 0]/np.linalg.norm(A[:, 0], 2)
    R[0, 0] = np.dot(A[:, 0].transpose(), Q[:, 0])
    for i in range(1, k):
        Q[:, i] = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, i].transpose(), Q[:, j])
            Q[:, i] = Q[:, i] - R[j,i]*Q[:, j]
        Q[:, i] = Q[:, i]/np.linalg.norm(Q[:, i], 2)
        R[i, i] = np.dot(A[:, i].transpose(), Q[:, i])
    if m<n :
        for i in range (k, n):
            for j in range(m):
                R[j, i] = np.dot(A[:, i].transpose(), Q[:, j])
    return Q, R


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A=QR
def householder_qr(A): # 7 pts
    # your code is here
    m, n = A.shape
    Q = np.eye(m, m)
    R = np.zeros((m, n))
    k = min(m, n)
    for i in range(k):
        x = A[i:, i]
        H = np.zeros((m, m))
        x = x.reshape(x.shape[0], 1)
        alpha = np.linalg.norm(x, 2)
        e = np.eye(m-i, 1)
        omega = np.zeros((m-i, 1))
        omega = x + alpha*e*np.sign(x[0])
        omega = omega/np.linalg.norm(omega, 2)
        if i == 0:
            H = np.eye(m, m)-2*np.dot(omega, omega.conj().transpose())
        else:
            H[:i, :i] = np.eye(i, i)
            H[i:m, i:m] = np.eye(m-i, m-i)-2*np.dot(omega, omega.conj().transpose())
        Q = np.dot(H, Q)
        A = np.dot(H, A)
        A[i+1:, i] = np.zeros(m-i-1)
    R = A
    return Q.T.conj(), R


# INPUT:  G - np.ndarray
# OUTPUT: A - np.ndarray (of size G.shape)
def pagerank_matrix(G): # 5 pts
    # enter your code here
    A = csr_matrix(G.shape)
    G = csr_matrix(G)
    L = np.array(1./G.sum(axis=1))
    L[np.isinf(L)] = 0
    L = L.reshape(len(L))
    B = diags(L, offsets=0)
    A = (G.T).dot(B)
    return csr_matrix(A)


# INPUT:  A - np.ndarray (2D), x0 - np.ndarray (1D), num_iter - integer (positive) 
# OUTPUT: x - np.ndarray (of size x0), l - float, res - np.ndarray (of size num_iter + 1 [include initial guess])
def power_method(A, x0, num_iter): # 5 pts
    # enter your code here
    x = x0/np.linalg.norm(x0, 2)
    res = np.zeros(num_iter+1)
    l = np.dot((A@x).T.conj(), x)
    res[0] = np.linalg.norm(A@x-l*x, 2)
    for i in range(1,num_iter+1):
        x = A@x
        x = x/np.linalg.norm(x, 2)
        l = np.dot((A@x).T.conj(), x)
        res[i] = np.linalg.norm(A@x-l*x, 2)
    return x, l, res


# INPUT:  A - np.ndarray (2D), d - float (from 0.0 to 1.0), x - np.ndarray (1D, size of A.shape[0/1])
# OUTPUT: y - np.ndarray (1D, size of x)
def pagerank_matvec(A, d, x): # 2 pts
    # enter your code here
    N = A.shape[0]
    y = (d*A.dot(x))+(1-d)*np.sum(x)*np.ones([N,1])/N
    return y


def return_words():
    # insert the (word, cosine_similarity) tuples
    # for the words 'numerical', 'linear', 'algebra' words from the notebook
    # into the corresponding lists below
    # words_and_cossim = [('word1', 'cossim1'), ...]
    numerical_words_and_cossim = [('computation', 0.547), ('mathematical', 0.532), ('calculations', 0.499), ('polynomial', 0.485),
                                  ('calculation', 0.473), ('practical', 0.46), ('statistical', 0.456), ('symbolic', 0.455), 
                                  ('geometric', 0.441), ('simplest', 0.438)]
                                                                                          
    linear_words_and_cossim = [('differential', 0.759), ('equations', 0.724), ('equation', 0.682), ('continuous', 0.674),
                               ('multiplication', 0.674), ('integral', 0.672), ('algebraic', 0.667), ('vector', 0.654), 
                               ('algebra', 0.63), ('inverse', 0.622)]
    algebra_words_and_cossim = [('geometry', 0.795), ('calculus', 0.73), ('algebraic', 0.716), ('differential', 0.687), 
                                ('equations', 0.665), ('equation', 0.648), ('theorem', 0.647), ('topology', 0.634), 
                                ('linear', 0.63), ('integral', 0.618)]
    
    return numerical_words_and_cossim, linear_words_and_cossim, algebra_words_and_cossim