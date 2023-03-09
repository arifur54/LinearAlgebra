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
    return Q, R

def backward_substitution(U, b):
    m, n = U.shape
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0.0:
            return None
        sum = 0.0
        for j in range(i + 1, n):
            sum += U[i][j] * x[j]
        x[i] = (b[i] - sum) / U[i][i]
    return x

if __name__ == "__main__":
    # A = np.array([[1, 0, 0, 0],
    #           [0, 1, 0, 0],
    #           [0, 0, 1, 0],
    #           [0, 0, 0, 1],
    #           [1, -1, 0, 0],
    #           [1, 0, -1, 0],
    #           [1, 0, 0, -1],
    #           [0, 1, -1, 0],
    #           [0, 1, 0, -1],
    #           [0, 0, 1, -1]])
    A = np.array([[-2, 1],
                  [-1, 2],
                  [1, -2],
                  [-1, -2]
                  ])
    
    x1 = 2.95
    x2 = 1.74
    x3 = -1.45
    x4 = 1.32

    # b = np.array([x1, x2, x3, x4, 1.23, 4.45, 1.61, 3.21, 0.45, -2.75])
    b = np.array([-1, 1, -1, -3])
    Q = np.array([[-0.7559,-0.0436],
                  [-0.378, 0.4364],
                  [0.378, -0.4364],
                  [-0.378, -0.7856] ])
    R = np.array([[2.6458,-1.5119],
                  [0, 3.2732]])

    # Q, R = qr_factorization(A)
    # print('R matrix after Householder transformation of matrix A displayed as .3 decimal places')
    # print(np.around(R, decimals=3), '\n')
    # print('Q matrix after Householder transformation of matrix A displayed as .3 decimal places')
    print(np.around(Q.T, decimals=3))

    # get the value of the altitudes
    b_hat = np.dot(Q.T, b)
    print(b_hat)
    x_hat = backward_substitution(R, b_hat)
    print("Best values of the altitudes/x_hat: \n", np.around(x_hat, decimals=3))

    # #The difference between the calculated values and the direct measurements
    # deltaX = np.array([x1, x2, x3, x4]) - x_hat
    # print("Difference between direct measurements and calculated values/deltaX: \n", np.around(deltaX, decimals=3)) 


