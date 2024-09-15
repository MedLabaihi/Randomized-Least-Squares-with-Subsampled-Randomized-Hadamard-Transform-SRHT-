import math as m
import numpy as np
from numpy import linalg
from scipy.linalg import hadamard


def Leverage_scores_exact(A):
    ''' Exact leverage score computation
        Input
            A: n-by-d matrix
        Output
            leverage_scores: Vector containing the leverage scores (squares of the L-2 norm of U(i,:)) where A = UΣV*, the SVD of A
    '''
    U, _, _ = linalg.svd(A, full_matrices=False)
    leverage_scores = np.linalg.norm(U, axis=1)**2  # Square of the 2-norm of each row of U
    return leverage_scores

def uniform_sketching(A, r):
    '''
    Uniform Sketching
    Input
        1. A: n-by-d dense matrix (n >> d)
        2. r: sketch size (number of rows to sample)
    Procedure
        1. Randomly sample r rows from A with uniform probability
        2. Optionally, scale the sampled matrix to adjust for sketching size
    Output
        sketch_A: r-by-d matrix (sketched version of A)
    '''
    n = A.shape[0]
    # Randomly choose r indices with replacement from the rows of A
    index = np.random.choice(n, r, replace=True)
    # Extract the rows indexed by the sampled indices and scale
    sketch_A = A[index, :] * np.sqrt(n / r)

    return sketch_A

def gaussian_projection(A, r):
    '''
    Random Gaussian Projection

    Input
        1. A: n-by-d dense matrix
        2. r: sketch size (number of rows in the sketching matrix)

    Procedure
        1. Construct an r-by-n sampling matrix S, where each entry of S is from a standard normal distribution
        2. Compute S*A to get the sketch

    Output
        sketch_A: r-by-d matrix (S*A)
        Here S is an r-by-n sketching matrix
    '''
    n = A.shape[0]
    # Create an r-by-n matrix with entries from a standard normal distribution
    S = np.random.randn(r, n) / np.sqrt(r)
    # Compute the matrix product S*A
    sketch_A = np.dot(S, A)

    return sketch_A

def uniform_SRHT(A, b, epsilon):
    '''
    Subsampled Randomized Hadamard Transform (Uniform)

    Input
        1. A: n-by-d dense matrix A
        2. b: n-by-1 dense vector b
        3. epsilon: error allowance

    Procedure:
        1. Compute Hadamard matrix H
        2. Normalize H
        3. Compute diagonal matrix D, the diagonal entries are independent Rademacher random variables
        4. Compute R, the sparsified random projection matrix using uniform distribution
        5. Compute RHD, the randomized Hadamard transform matrix
        6. Compute RHD*A and RHD*b

    Output
        1. A_hat: The sketched r-by-d matrix
        2. SRHT: r-by-n sampling matrix
        3. b: The vector b, unchanged if n is a power of 2
    '''
    n = A.shape[0]
    d = A.shape[1]

    if epsilon <= 1:
        r = int(np.ceil(d * (np.log(n) ** 3) / epsilon ** 2))
    else:
        r = int(np.ceil(d * (np.log(n) ** 3)))

    if r >= n:
        raise ValueError("No reduction of dimensionality, enter a new epsilon")

    n_hat = int(2 ** np.ceil(np.log2(n)))
    res = n_hat - n

    A_padded = np.vstack([A, np.zeros((res, d))])
    b_padded = np.concatenate([b, np.zeros((res,))])

    H = hadamard(n_hat) / np.sqrt(n_hat)
    D = np.diag(np.random.choice([-1, 1], size=n_hat))

    R = np.zeros((r, n_hat))
    k = np.random.choice(n_hat, r, replace=True)
    for i in range(r):
        R[i, k[i]] = 1 / np.sqrt(r / n)

    RHD = np.matmul(np.matmul(R, H), D)
    A_hat = np.matmul(RHD, A_padded)

    return A_hat, RHD, b_padded

def SRHT(A, b, epsilon):
    '''
    Subsampled Randomized Hadamard Transform (Non-Uniform)

    Input
        1. A: n-by-d dense matrix A
        2. b: n-by-1 dense vector b
        3. epsilon: error allowance

    Procedure:
        1. Compute Hadamard matrix H
        2. Normalize H
        3. Compute diagonal matrix D with independent Rademacher random variables
        4. Compute R, the sparsified random projection matrix using leverage probability sampling
        5. Compute RHD, the randomized Hadamard transform matrix
        6. Compute RHD*A and RHD*b

    Output
        1. A_hat: The sketched r-by-d matrix
        2. SRHT: r-by-n sampling matrix
        3. b: The vector b, unchanged if n is a power of 2
    '''
    n = A.shape[0]
    d = A.shape[1]

    if epsilon <= 1:
        r = int(np.ceil(d * (np.log(n) ** 3) / epsilon ** 2))
    else:
        r = int(np.ceil(d * (np.log(n) ** 3)))

    if r >= n:
        raise ValueError("No reduction of dimensionality, enter a new epsilon")

    n_hat = int(2 ** np.ceil(np.log2(n)))
    res = n_hat - n

    # Padding A and b
    A_padded = np.vstack([A, np.zeros((res, d))])
    b_padded = np.concatenate([b, np.zeros((res,))])

    # Compute Hadamard matrix and normalize
    H = hadamard(n_hat) / np.sqrt(n_hat)

    # Compute diagonal matrix D
    D = np.diag(np.random.choice([-1, 1], size=n_hat))

    # Compute leverage scores and probabilities
    leverage_scores = Leverage_scores_exact(A)
    leverage_probabilities = leverage_scores / np.sum(leverage_scores)

    # Construct the R matrix
    R = np.zeros((r, n_hat))
    k = np.random.choice(n_hat, r, replace=True, p=leverage_probabilities)
    for i in range(r):
        R[i, k[i]] = 1 / np.sqrt(leverage_probabilities[k[i]] * r)

    # Compute RHD and the sketched matrix A_hat
    RHD = np.matmul(np.matmul(R, H), D)
    A_hat = np.matmul(RHD, A_padded)

    return A_hat, RHD, b_padded
