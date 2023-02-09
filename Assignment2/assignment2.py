def generateHb(n):
    H = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(1/(i + j + 1))
        H.append(row)
        
    x = [1 for i in range(n)]
    b = [0 for i in range(n)]
    for i in range(n):
        for j in range(n):
            b[i] += H[i][j] * x[j]
    
    return (H, b)

def forward_substitution(L, b):
    n = len(b)
    x = [[0.0] * n for i in range(n)]
    for i in range(n):
        if L[i][i] == 0.0:
            return None
        sum = 0
        for j in range(i):
            sum += L[i][j] * x[j]
        x[i] = (b[i] - sum) / L[i][i]
    return x

def backward_substitution(U, b):
    n = len(b)
    x = [[0.0] * n for i in range(n)]
    for i in range(n - 1, -1, -1):
        if U[i][i] == 0.0:
            return None
        sum = 0.0
        for j in range(i + 1, n):
            sum += U[i][j] * x[j]
        x[i] = (b[i] - sum) / U[i][i]
    return x

def upper_triangular_matrix_U(k, n, L, U, A):
    # Get the upper triangular matrix U
    for j in range(k, n):
        sum = 0.0
        for p in range(k):
            sum += L[k][p] * U[p][j]
        U[k][j] = A[k][j] - sum
    print(f"upper triangular matrix step {k+1}: {U}")

def lower_triangular_matrix_L(k, n, L, U, A):
    # Get the lower triangular matrix L
    for i in range(k + 1, n):
        sum = 0.0
        for p in range(k):
            sum += L[i][p] * U[p][k]
        L[i][k] = (A[i][k] - sum) / U[k][k]
    print(f"Lower triangular matrix step {k+1}: {L}\n")

def LU_decomposition(A):
    n = len(A)
    L = [[0.0] * n for i in range(n)]
    U = [[0.0] * n for i in range(n)]

    for k in range(n):
        if A[k][k] == 0:
            return None

        L[k][k] = 1
        upper_triangular_matrix_U(k, n, L, U, A)
        lower_triangular_matrix_L(k, n, L, U, A)

    return L, U



def gauss_elimination(A, b):
    L, U = LU_decomposition(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x



if __name__ == '__main__':
    # # Lower Triangle Matrix Forward Subtitution
    # print("=================== Forward_Subtitution ===============================================")
    # L = [[4,0,0],[2,-2,0],[1,3,4]]
    # b =[1, -2, 19]
    # res = forward_substitution(L, b)
    # print(f"L = {L}")
    # print(f"b = {b}")
    # print(f"forward subtitution result = {res}")

    # # Upper Triangle Matrix Backward_subtitution
    # print("=================== Backward_subtitution ===============================================")
    # U = [[1, 3, 4], [0, -2, 2], [0, 0, 4]]
    # b = [11, -2, 4]
    # res = backward_substitution(U, b)
    # print(f"U = {U}")
    # print(f"b = {b}")
    # print(f"forward subtitution result = {res}")

    # # LU_Decomposition
    # print("=================== LU_Decomposition ===============================================")
    # A = [[4, 1, 2], [3, 4, 1], [2, 1, 3]]
    # result = LU_decomposition(A)
    # if result is None:
    #     print("Matrix is singular.")
    # else:
    #     L, U = result
    #     print(f"L: {L[0]}\n {L[1]}\n {L[2]}")
    #     print(f"L: {U[0]}\n {U[1]}\n {U[2]}")

    # # gauss_elimination
    # print("=================== gauss_elimination ===============================================")
    
    # A = [[1, 2, 1, -1], [3, 2, 4, 4], [4, 4, 3, 4], [2, 0, 1, 5]]
    # b = [5, 16, 22, 15]
    # res = gauss_elimination(A, b)
    # print(f"A = {A}")
    # print(f"b = {b}")
    # print(f"Guass Elimination Solution = {res}")

    # Generate Hilbert Matrix
    print("=================== Generate Hilbert Matrix ===============================================")

    n = 20
    H, b = generateHb(n)
    print("Hilbert Matrix for n =", n)
    for row in H:
        print(row)
    print("\nVector b = Hx for n =", n)
    print(b)

    x_hat = gauss_elimination(H, b)
    # A, b = forward_substitution(A, b)
    # x_hat = backward_substitution(A, b)

    print(x_hat)





