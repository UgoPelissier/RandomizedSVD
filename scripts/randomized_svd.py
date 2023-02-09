# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:41:09 2023

@author: ugo.pelissier
"""

from matplotlib.image import imread
from tabulate import tabulate
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from scipy.linalg import interpolative, rq

#-----------------------------------------------------------------------------#
# RANDOMIZED SVD
#-----------------------------------------------------------------------------#
def rSVD(A,r,q,p):
    """

    Parameters
    ----------
    A : numpy array
        Target matrix.
    r : int
        Target rank.
    q : int
        Power iterarions.
    p : int
        Oversampling factor.

    Returns
    -------
    Low-rank SVD approximation for A.

    """
    start = time.time()
    
    # Step 1 : Sample column space of A with random matrix
    m, n = A.shape
    l=r+p
    Omega = np.random.randn(n,l)
    Y = A @ Omega
    for k in range(q):
        Y = A @ (A.T @ Y)      
    Q, R = np.linalg.qr(Y,mode='reduced')
    
    # Step 2: Compute SVD on projected Y = Q.T @ A
    Z = Q.T @ A
    UY, S, VT = np.linalg.svd(Z,full_matrices=0)
    U = Q @ UY
    
    end = time.time()
    
    return U, S, VT, (end-start)

def adaptative_randomized_range_finder(A, r, eps, max_iter):
    """

    Parameters
    ----------
    A : numpy array
        Target matrix.
    r : int
        Size of the standard gaussian vector used for error approximation.
    eps : float
        Tolerance for error approximation.
    max_iter : int
        Max number of iterations.

    Returns
    -------
    Q : numpy array
        Orthonormal matrix whose range approximates the range of X.
    approx_error_list : array of floats
        DESCRIPTION.
    true_error_list : array of floats
        DESCRIPTION.
    j : int
        Number of iterations.

    """
    m, n = A.shape
    omega = np.random.randn(n,r) # Gaussian vectors
    y = A @ omega
    
    # Stoping criteria
    threshold = eps/(10*np.sqrt(2/np.pi))
    approx_error = np.linalg.norm(y[:,-r:],axis=0).max()
    
    # Initialization
    Q = np.empty([m, 0])
    I = np.eye(m)
    approx_error_list = []
    true_error_list = []
    
    for j in range(max_iter):
        if (approx_error<threshold):
            # print("\nAdaptative randomized range finder converged in {} iterations.".format(j))
            return Q, np.array(approx_error_list), np.array(true_error_list), j
        
        else:
            
            # Project y onto Q
            y[:,j] = (I - Q @ Q.T) @ y[:, j]
            
            # Normalize and append
            q = y[:,j] / np.linalg.norm(y[:,j])
            Q = np.concatenate((Q, q.reshape(len(q),1)), axis=1)
            
            # New gaussian vector
            omega = np.random.randn(n,1)
            
            # Get its approximation error
            y_new = (I - Q @ Q.T) @ (A @ omega)
            
            # Append to y
            y = np.concatenate((y, y_new), axis=1)
            
            # Update previous y(i)
            for k in range(j+1,j+r):
                y[:,k] -= q*np.dot(q, y[:,k])
            
            # Compute new approx error
            approx_error = np.linalg.norm(y[:,-r:],axis=0).max()
            approx_error_list.append(approx_error)
            
            # True error (normally not computed)
            true_error = np.linalg.norm(((I - Q @ Q.T) @ A),ord=2)
            true_error_list.append(true_error)
    
    print("Maximum iteration reached. Consider increase max_iter.")
    
    return Q, np.array(approx_error_list), np.array(true_error_list), j

def adaptative_range_finder_rSVD(A, r, eps, max_iter, q, p):
    """

    Parameters
    ----------
    A : numpy array
        Target matrix.
    r : int
        Size of the standard gaussian vector used for error approximation.
    eps : float
        Tolerance for error approximation.
    max_iter : int
        Max number of iterations.
    q : int
        Power iterarions.
    p : int
        Oversampling factor.
        
    Returns
    -------
    Low-rank SVD approximation for X.

    """
    start = time.time()
    
    # Step 1 : Sample column space of X with random P matrix   
    Q, _, _, _ = adaptative_randomized_range_finder(A, r, eps, max_iter)
    
    # Step 2: Compute SVD on projected Y = Q.T @ A
    Y = Q.T @ A
    UY, S, VT = np.linalg.svd(Y,full_matrices=0)
    U = Q @ UY
    
    end = time.time()
    
    return U, S, VT, (end-start)

def randomized_subspace_iteration(A,r,q,p):
    """

    Parameters
    ----------
    A : numpy array
        Target matrix.
    r : int
        Target rank.
    q : int
        Power iterarions.
    p : int
        Oversampling factor.

    Returns
    -------
    

    """
    l=r+p
    m, n = A.shape
    omega = np.random.randn(n,l) # Gaussian vectors
    
    Y = A @ omega
    Q, R = np.linalg.qr(Y,mode='reduced')
    
    for j in range(1,q+1):
        Y_tilde = A.T @ Q
        Q_tilde, R_tilde = np.linalg.qr(Y_tilde,mode='reduced')
        
        Y = A @ Q_tilde
        Q, R = np.linalg.qr(Y,mode='reduced')
    
    return Q

def randomized_subspace_iteration_rSVD(A,r,q,p):
    """

    Parameters
    ----------
    A : numpy array
        Target matrix.
    r : int
        Target rank.
    q : int
        Power iterarions.
    p : int
        Oversampling factor.

    Returns
    -------
    

    """
    start = time.time()
    
    # Step 1 : Sample column space of A with random P matrix   
    Q = randomized_subspace_iteration(A,r,q,p)
    
    # Step 2: Compute SVD on projected Y = Q.T @ A
    Y = Q.T @ A
    UY, S, VT = np.linalg.svd(Y,full_matrices=0)
    U = Q @ UY
    
    end = time.time()
    
    return U, S, VT, (end-start)

def row_extraction_svd(A,r,q,p):
    """

    Parameters
    ----------
    A : numpy array
        Target matrix.
    r : int
        Target rank.
    q : int
        Power iterarions.
    p : int
        Oversampling factor.

    Returns
    -------
    Low-rank SVD approximation for A.

    """
    start = time.time()
    
    # Step 1 : Sample column space of A with random matrix
    m, n = A.shape
    l=r+p
    Omega = np.random.randn(n,l)
    Y = A @ Omega
    for k in range(q):
        Y = A @ (A.T @ Y)
        
    Q, R = np.linalg.qr(Y,mode='reduced')
    J, proj = interpolative.interp_decomp(Q.T, l, rand=True)
    
    X = np.zeros((Q.T).shape)
    X[:,J[:l]] = np.eye(l)
    X[:,J[l:]] = proj
    
    # R, W = rq(A[J[:l],:], mode='economic')
    # Z = X.T @ R
    # U, S, VT_tilde = np.linalg.svd(Z,full_matrices=0)
    # V = W.T @ VT_tilde.T
    
    # end = time.time()
    
    # return U, S, V.T, (end-start)
    UY, S, VT = np.linalg.svd(A[J[:l],:],full_matrices=0)
    U = X.T @ UY
    
    end = time.time()
    return U, S, VT, (end-start)
    
    
def reconstruction(A, r, eps, max_iter, target_rank, q, p):
    """

    Parameters
    ----------
    A : numpy array
        Target matrix.
    r : int
        Size of the standard gaussian vector used for error approximation.
    eps : float
        Tolerance for error approximation.
    max_iter : int
        Max number of iterations.
    target_rank : int
        Target rank for classic randomized SVD.
    q : int
        Power iterarions.
    p : int
        Oversampling factor.
        
    Returns
    -------
    SVD decomposition, error and computation time for classic SVD, randomized SVD, and adaptative randomized SVD

    """
    start = time.time()
    U, S, VT = np.linalg.svd(A) # Full SVD decomposition
    end = time.time()
    ASVD = U[:,:(target_rank+1)] @ np.diag(S[:(target_rank+1)]) @ VT[:(target_rank+1),:] # SVD approximation
    epsSVD = np.linalg.norm(A-ASVD)/np.linalg.norm(A)

    rU, rS, rVT, rt = rSVD(A, target_rank, q, p)
    ArSVD = rU[:,:(target_rank+1)] @ np.diag(rS[:(target_rank+1)]) @ rVT[:(target_rank+1),:] # SVD approximation
    epsrSVD = np.linalg.norm(A-ArSVD)/np.linalg.norm(A)
    
    # adaptative_rU, adaptative_rS, adaptative_rVT, adaptative_rt = adaptative_range_finder_rSVD(A, r, eps, max_iter, q, p)
    # adaptative_ArSVD = adaptative_rU[:,:(r+1)] @ np.diag(adaptative_rS[:(r+1)]) @ adaptative_rVT[:(r+1),:] # SVD approximation
    # eps_adaptative_rSVD = np.linalg.norm(A-adaptative_ArSVD)/np.linalg.norm(A)
    
    # iteration_rU, iteration_rS, iteration_rVT, iteration_rt = randomized_subspace_iteration_rSVD(A, target_rank, q, p)
    # iteration_ArSVD = iteration_rU[:,:(r+1)] @ np.diag(iteration_rS[:(r+1)]) @ iteration_rVT[:(r+1),:] # SVD approximation
    # eps_iteration_rSVD = np.linalg.norm(A-iteration_ArSVD)/np.linalg.norm(A)
    
    row_extraction_rU, row_extraction_rS, row_extraction_rVT, row_extraction_rt = row_extraction_svd(A, target_rank, q, p)
    row_extraction_ArSVD = row_extraction_rU[:,:(target_rank+1)] @ np.diag(row_extraction_rS[:(target_rank+1)]) @ row_extraction_rVT[:(target_rank+1),:] # SVD approximation
    eps_row_extraction_rSVD = np.linalg.norm(A-row_extraction_ArSVD)/np.linalg.norm(A)
    
    return ASVD, ArSVD, row_extraction_ArSVD, epsSVD, epsrSVD, eps_row_extraction_rSVD, (end-start), rt, row_extraction_rt

#-----------------------------------------------------------------------------#
# MATRIX GENERATORS
#-----------------------------------------------------------------------------#
def fixed_rank_matrix(m, n, rank):
    """

    Parameters
    ----------
    m : int
        Number of rows.
    n : TYPE
        Number of columns.
    rank : int
        Rank of the matrix.

    Returns
    -------
    A : numpy array
        Matrix of size (m,n) & rank=r.
    S : numpy array
        Singular values.

    """
    A = np.random.rand(m,n)
    U, S, VT = np.linalg.svd(A,full_matrices=0)
    S[rank:]=0
    S = (S/S.max())**3  # quickly decaying spectrum
    A = U @ np.diag(S) @ VT
    return A, S

def linear_decaying_spectrum(m, n):
    """

    Parameters
    ----------
    m : int
        Number of rows.
    n : int
        Number of columns

    Returns
    -------
    A : numpy array
        Matrix of size (m,n).
    S : numpy array
        Singular values.

    """
    A = np.random.rand(m,n)
    U, S, VT = np.linalg.svd(A,full_matrices=0)
    S[:m] = np.arange(1,0,-1/n)
    A = U @ np.diag(S) @ VT
    return A, S

#-----------------------------------------------------------------------------#
# POST-PROCESS
#-----------------------------------------------------------------------------#
def time_compare(A, p, q):
    start = time.time()
    U, S, VT = np.linalg.svd(A) # Full SVD decomposition
    end = time.time()
    t_SVD = end-start
        
    X = []
    T_rSVD = []
    T_extract_rSVD = []
    for i in range(1,501,10):
        X.append(i)
        
        rU, rS, rVT, rt = rSVD(A, i, q, p)
        T_rSVD.append(rt)
        
        row_extraction_rU, row_extraction_rS, row_extraction_rVT, row_extraction_rt = row_extraction_svd(A, i, q, p)
        T_extract_rSVD.append(row_extraction_rt)
        
    T_SVD = [t_SVD for i in range(len(T_rSVD))]
    
    plt.semilogy(X,T_SVD,'r',label='SVD')
    plt.semilogy(X,T_rSVD,'k',label='rSVD')
    plt.semilogy(X,T_extract_rSVD,'b',label='row_extract_rSVD')
    
    plt.xlabel("Target rank")
    plt.ylabel("Time (s)")
    
    plt.legend()
    plt.savefig('time_compare.png', dpi=600)
    plt.show()
    
    
def check_adaptative_randomized_range_finder(X, m, n, rank, r, eps, max_iter):
    A, S = fixed_rank_matrix(m, n, rank)
    Q_approx, approx_error_list, true_error_list, j = adaptative_randomized_range_finder(A, r, eps, max_iter)
    
    print(f"\nReal rank: {rank} | Computed rank: {Q_approx.shape[1]}")
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(approx_error_list, 'k', label="estimated error")
    ax.plot(true_error_list, 'r', label="true error")
    ax.set(yscale='log', ylabel="error", xlabel='iter', title=f"Real rank: {rank} | Computed rank: {Q_approx.shape[1]}")
    plt.legend()
    plt.savefig('adaptative_randomized_range_finder.png', dpi=600)
    plt.show()
    
    Q_approx, approx_error_list, true_error_list, j = adaptative_randomized_range_finder(X, r, eps, max_iter)
    fig, ax = plt.subplots(1, 1)
    ax.plot(approx_error_list, 'k', label="estimated error")
    ax.plot(true_error_list, 'r', label="true error")
    ax.set(yscale='log', ylabel="error", xlabel='iter', title=f"Computed rank: {Q_approx.shape[1]}")
    plt.legend()
    plt.savefig('adaptative_randomized_range_finder.png', dpi=600)
    plt.show()
    
def accuracy_check(A, q, p):
    step = 50
    start = 1
    
    m, n = A.shape
    
    U, S, VT = np.linalg.svd(A,full_matrices=0) #SVD decompisition
    
    error = []
    for r in range(start,start+step):
        rU, rS, rVT, rt = rSVD(A, r, q=0, p=0)
        ArSVD = rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:] # SVD approximation
        epsrSVD = np.linalg.norm(A-ArSVD,ord=2)
        error.append(epsrSVD)
    plt.plot(np.log10(error), 'b-x', label="$log_{10}(e_{k})$ - $(q,p)=(0,0)$")
        
    error = []
    for r in range(start,start+step):
        rU, rS, rVT, rt = rSVD(A, r, q=0, p=5)
        ArSVD = rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:] # SVD approximation
        epsrSVD = np.linalg.norm(A-ArSVD,ord=2)
        error.append(epsrSVD)
    plt.plot(np.log10(error), 'r-x', label="$log_{10}(e_{k})$ - $(q,p)=(0,5)$")
    
    error = []
    for r in range(start,start+step):
        rU, rS, rVT, rt = rSVD(A, r, q=2, p=5)
        ArSVD = rU[:,:(r+1)] @ np.diag(rS[:(r+1)]) @ rVT[:(r+1),:] # SVD approximation
        epsrSVD = np.linalg.norm(A-ArSVD,ord=2)
        error.append(epsrSVD)
    plt.plot(np.log10(error), 'g-x', label="$log_{10}(e_{k})$ - $(q,p)=(2,5)$")
    
    plt.plot(np.log10(S[start+1:start+step+1]), 'k-x', label="$log_{10}(\sigma_{k+1})$")
    
    plt.xlabel("$k$")
    plt.legend()
    plt.savefig('accuracy.png', dpi=600)
    plt.show()
    
def plot_singular_values(S):
    plt.semilogy(np.diag(S))
    plt.title('Singular values')
    plt.show()

def plot_images(ASVD, ArSVD, row_extraction_ArSVD, target_rank):
    fig, axs = plt.subplots(1,3)

    plt.set_cmap('gray')

    axs[0].imshow(ASVD)
    axs[0].axis('off')
    axs[0].title.set_text('SVD')
    axs[1].imshow(ArSVD)
    axs[1].axis('off')
    axs[1].title.set_text('rSVD')
    axs[2].imshow(row_extraction_ArSVD)
    axs[2].axis('off')
    axs[2].title.set_text('Row extract rSVD')
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    fig.savefig(f"jupiter_svd_compare_r_{target_rank}.png", dpi=600)
    plt.show()
    
def plot_compare(A, r, eps, max_iter, target_rank, q, p):
    ASVD, ArSVD, row_extraction_ArSVD, epsSVD, epsrSVD, eps_row_extraction_rSVD, t, rt, row_extraction_rt = reconstruction(A, r=10, eps=1e-5, max_iter=A.shape[1], target_rank=target_rank, q=q, p=p)
    plot_images(ASVD, ArSVD, row_extraction_ArSVD, target_rank)
    
def illustrate_power_iterations():
    A, S = linear_decaying_spectrum(m=1000, n=100)
    
    color_list = np.array([[0,0,2/3],
                           [0,0,1],
                           [0,1/3,1],
                           [0,2/3,1],
                           [0,1,1],
                           [1/3,1,2/3],
                           [2/3,1,1/3],
                           [1,1,0],
                           [1,2/3,0],
                           [1,1/3,0],
                           [1,0,0],
                           [2/3,0,0]])
    
    plt.plot(S,color='k',label='SVD')
    
    Y = A
    for q in range(1,6):
        Y = A.T @ Y
        Y = A @ Y
        Uq, Sq, VTq = np.linalg.svd(Y,full_matrices=0)
        plt.plot(Sq, color=tuple(color_list[2*q+1]),label='rSVD, q='+str(q))
    
    plt.xlabel("$k$")
    plt.legend()
    plt.savefig('power_iteration.png', dpi=600)
    plt.show()
    
def printInfo(epsSVD, epsrSVD, eps_row_extraction_rSVD, t, rt, row_extraction_rt):
    data = [["Error",epsSVD, epsrSVD, eps_row_extraction_rSVD],
            ["Time (s)",t, rt, row_extraction_rt]]
    print ("\n",tabulate(data, headers=["","SVD","rSVD", "row_extract_rSVD"]))
    

A = imread('data/jupiter.png')
A = np.mean(A,axis=2) # Convert RGB to grayscale

# img = plt.imshow(A)
# img.set_cmap('gray')
# plt.axis('off')
# plt.show()

target_rank = 1000
q = 2
p = 5

ASVD, ArSVD, row_extraction_ArSVD, epsSVD, epsrSVD, eps_row_extraction_rSVD, t, rt, row_extraction_rt = reconstruction(A, r=10, eps=1e-5, max_iter=A.shape[1], target_rank=target_rank, q=q, p=p)

# printInfo(epsSVD, epsrSVD, eps_row_extraction_rSVD, t, rt, row_extraction_rt)

# check_adaptative_randomized_range_finder(m=1000, n=1000, rank=50, r=20, eps=1e-5, max_iter=200)

# illustrate_power_iterations()

# accuracy_check(A, q, p)

# check_adaptative_randomized_range_finder(A, m=1000, n=500, rank=50, r=10, eps=1e-5, max_iter=1600)

# time_compare(A, p, q)

plot_compare(A, r=10, eps=1e-5, max_iter=A.shape[1], target_rank=target_rank, q=q, p=p)