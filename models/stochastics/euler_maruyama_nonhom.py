import numpy as np
from numba import njit, prange

# Parameters
T = 5.0
N = 10000
Wi = 50
delta_t = 1/1600
b = 30.0
d = 2.0
C = (b + d + 2) / b * 1 / Wi
M = int(T / delta_t)

@njit(fastmath=True)
def F(Q):
    norm_Q = np.sqrt(np.sum(Q**2))
    return Q / (1 - (norm_Q**2 / b))

@njit(fastmath=True)
def velocity_gradient(y):
    grad = np.zeros((2, 2), dtype=np.float64)
    grad[0, 0] = 1 - 2 * y
    grad[1, 0] = 0
    return grad

@njit(fastmath=True, parallel=True)
def euler_maruyama_nonhomogeneous(Q0, delta_t, T, random_numbers, Wi, b):
    N, D = Q0.shape  
    M = int(T / delta_t)
    Q_traj = np.zeros((N, M + 1, D))
    Q_traj[:, 0, :] = Q0
    sqrt_delta_t = np.sqrt(delta_t)
    sqrt_1_Wi = np.sqrt(1 / Wi)

    for i in prange(N):
        for n in range(M):
            y_position = Q_traj[i, n, 1]
            kappa_T = velocity_gradient(y_position)
            dW = random_numbers[i, n, :] * sqrt_delta_t
            F_Q = F(Q_traj[i, n, :])
            Q_dot = np.dot(kappa_T, Q_traj[i, n, :]) - (1 / (2 * Wi)) * F_Q
            Q_traj[i, n + 1, :] = Q_traj[i, n, :] + Q_dot * delta_t + sqrt_1_Wi * dW

    return Q_traj

@njit(fastmath=True, parallel=True)
def compute_polymer_stress_tensor(Q_traj_samples, N, M, Wi, b, C):
    tau_np = np.zeros((2, 2, M + 1))
    for n in prange(M + 1):
        sum_dyadic_product = np.zeros((2, 2))
        for I in range(N):
            Q = Q_traj_samples[I, n, :]
            F_Q = F(Q)
            sum_dyadic_product += np.outer(Q, F_Q)
        tau_np[:, :, n] = C * (sum_dyadic_product / N)
    return tau_np

np.random.seed(42)
Q0 = np.random.normal(0.0, 0.1, (N, 2))
random_numbers = np.random.normal(size=(N, M, 2))

# Execute functions
Q_traj_samples = euler_maruyama_nonhomogeneous(Q0, delta_t, T, random_numbers, Wi, b)
tau_np = compute_polymer_stress_tensor(Q_traj_samples, N, M, Wi, b, C)
print("Polymer Stress Tensor over Time:\n", tau_np)
