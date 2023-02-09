# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 11:22:57 2023

@author: ugo.pelissier
"""

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

A = imread('data/jupiter.png')
X = np.mean(A,axis=2) # Convert RGB to grayscale

img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()

U, S, VT = np.linalg.svd(X,full_matrices=False)
S = np.diag(S)

j = 0
for r in (2, 20, 100):
    # Construct approximate image
    Xapprox = U[:,:r] @ S[0:r,0:r] @ VT[:r,:]
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    plt.show()


# SVD analysis
plt.semilogy(np.diag(S[:1000]),'k')
plt.xlabel('$k$')
plt.title('Singular values')
plt.savefig('singular_values.png', dpi=600)
plt.show()

plt.plot(np.cumsum(np.diag(S[:1000]))/np.sum(np.diag(S)),'k')
plt.xlabel('$k$')
plt.title('Singular values: Cumulative Sum')
plt.savefig('cumulative_sum.png', dpi=600)
plt.show()