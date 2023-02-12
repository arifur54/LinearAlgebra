import numpy as np

def householder(A):
    m, n = A.shape
    R = np.copy(A)
    Q = np.identity(m)
    for i in range(n):
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        H = np.identity(m)
        H[i:, i:] -= 2 * np.outer(v, v)
        R = np.dot(H, R)
        Q = np.dot(Q, np.transpose(H))
        print(f"Step ======= {i} =========")
        print(f"Q: {Q} \n R: {R}")
    return Q, R

A = np.array([[1, -1, 4], [1, 4, -2], [1, 4, 2], [1, -1, 0]])
Q, R = householder(A)

print(f"A:\n {A}")
print(f"Q Result :\n {Q}")
print(f"R Result :\n {R}")

print(f"Varify Result (multiply Q * R) -> \n got: \n {np.dot(Q, R)} \n given: \n {A}")
