import numpy as np

def column_convertor(x):
    x.shape = (1, x.shape[0])
    return x

def get_norm(x):
    return np.sqrt(np.sum(np.square(x)))

def householder_transformation(v):
    vector_size = v.shape[1]
    e = np.zeros_like(v)
    e[0, 0] = 1
    vector = get_norm(v) * e
    if v[0,0] < 0:
        vector = - vector
    updatedV = (v + vector).astype(np.float32)
    H = np.identity(vector_size) - ((2 * np.matmul(np.transpose(updatedV), updatedV)) / np.matmul(updatedV, np.transpose(updatedV)))
    return H

def qr_factorization(A):
    n, m = A.shape
    Q = np.identity(n)
    R = A.astype(np.float32)
    for i in range(min(n, m)):
        v = column_convertor(R[i:, i])
        Hbar = householder_transformation(v)
        H = np.identity(n)
        H[i:, i:] = Hbar
        R = np.matmul(H, R)
        Q = np.matmul(Q, H)
        R = np.around(R, decimals=5)
        Q = np.around(Q, decimals=5)
        print(f"Step ======= {i+1} =========")
        print(f"Q: {Q} \n R: {R}")
       
    return Q, R

if __name__ == "__main__":
    A = np.array([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
    Q, R = qr_factorization(A)
    R = np.around(R, decimals=5)
    Q = np.around(Q, decimals=5)
    print('A after QR factorization')
    print('R matrix')
    print(R, '\n')
    print('Q matrix')
    print(Q)


