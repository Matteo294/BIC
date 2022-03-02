import numpy as np
from matplotlib import pyplot as plt 

N1 = 9
N2 = 5
N3 = 3
N4 = 2
L = 1
D = 0.5
h = 2*L/(N1-1)
eps = 1
nrelax = 5
phi0 = 1
ncycles = 20

def Jacobi_Iteration(x, Dinv, L, U, b):
    #print(x.shape, Dinv.shape, L.shape, U.shape, b.shape)
    return Dinv*b + Dinv*np.einsum("ij,j", L+U, x)

# Matrix to move from 2h to h
def create_restr_mat(N):
    assert(N%2 == 1)
    Nrestr = int((N-1)/2) + 1
    R = np.zeros((Nrestr, N))
    for i in range(1, Nrestr-1):
        R[i,2*i-1] = 1
        R[i,2*i] = 2
        R[i,2*i+1] = 1
    R[0][0] = 2
    R[0][1] = 1
    R[Nrestr-1][2*(Nrestr-1)] = 2
    R[Nrestr-1][2*(Nrestr-1)-1] = 1
    return R

# Returns M1*A*M2
def restrict_prolong_matrix(A, M1, M2):
    return np.einsum("ij,jk,kl->il", M1, A, M2)
# Returns M*v
def mat_vec_prod(M, v):
    return np.einsum("ij,j", M, v)

# Returns D^-1, L, U given matrix A
def Dinv_L_U_decomposition(A):
    D_inv = np.zeros(len(A))
    L = np.zeros((len(A), len(A)))
    U = np.zeros((len(A), len(A)))
    for i in range(len(A)):
        for j in range(len(A)):
            if i < j:
                L[i,j] = -A[i,j]
            elif i > j:
                U[i,j] = -A[i,j]
            elif i == j:
                D_inv[i] = 1/A[i,i]
    return D_inv, L, U 

def solve_Vcycle(x, A, b):
    if len(x) == 2:
         # Decompose in to D L U
        D_inv, L, U = Dinv_L_U_decomposition(A)
        # One relaxation step 
        x = Jacobi_Iteration(x, D_inv, L, U, b)
    else:
        # Decompose in to D L U
        D_inv, L, U = Dinv_L_U_decomposition(A)
        # One relaxation step 
        x = Jacobi_Iteration(x, D_inv, L, U, b)
        # Compute residual
        residual = b - mat_vec_prod(A, x)
        # Create restriction and prolongation matrices
        R = create_restr_mat(len(residual))
        P = 2*np.transpose(R)
        # Restrict things
        residual_r = mat_vec_prod(R, residual)
        mat_vec_prod(R, b)
        A_r = restrict_prolong_matrix(A, R, P)   
        # Solver A*err = res
        err = np.zeros(len(residual_r))
        err = solve_Vcycle(err, A_r, residual_r)
        # Prolong error and correct
        err_p = mat_vec_prod(P, err)
        x = x + err_p
        # Further relaxation
        x = Jacobi_Iteration(x, D_inv, L, U, b)
    return x

A = np.zeros((N1, N1))
for i in range(1, N1-1):
    A[i,i-1] = 1
    A[i,i] = -2
    A[i,i+1] = 1
A[0][0] = 1
A[N1-1][N1-1] = 1
b = np.full(N1, -h**2/D/eps)
b[0] = phi0
b[N1-1] = phi0
x = np.zeros(N1)

# Solve
solutions = [[] for _ in range(ncycles)]
print("Starting v-cycle")
for n in range(ncycles):
    x = solve_Vcycle(x, A, b)
    print("Residuals:", b - mat_vec_prod(A, x))
    solutions[n] = x

# Plotting
gridpoints = np.linspace(-L, L, N1)
for n in range(ncycles):
    plt.plot(gridpoints, solutions[n])
plt.show()
