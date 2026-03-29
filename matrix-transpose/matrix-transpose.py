import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    n, m = A.shape
    out = np.empty((m, n), dtype=A.dtype)
    for i in range(n):
        for j in range(m):
            out[j, i] = A[i, j]
    return out
    pass
