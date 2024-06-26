import numpy as np
from numba import njit, prange

# Parameters for the SDE
T = 5.0 
N = 10000
Wi = 50
sigma = 1.0
kappa = np.array([[sigma, 0.0], [0.0, -sigma]])
Q0_mean = np.array([0.0, 0.0])
Q0_cov = np.array([[0.1, 0.0], [0.0, 0.1]])
delta_t = 1/1600
b = 30.0 
d = 2.0
gamma = 1.0
ReWi = 1.0
C = (b + d + 2) / b * 1 / Wi

@njit
def F(Q):
    norm_Q = np.sqrt(np.sum(Q**2))
    return Q / (1 - (norm_Q**2 / b))

@njit(fastmath=True, parallel=True)
def euler_maruyama_vectorized(Q0, kappa, Wi, delta_t, T, random_numbers):
    N, D = Q0.shape  
    M = int(T / delta_t)
    Q_traj = np.zeros((N, M + 1, D))
    Q_traj[:, 0, :] = Q0
    sqrt_delta_t = np.sqrt(delta_t)
    sqrt_1_Wi = np.sqrt(1 / Wi)
    kappa_T = np.ascontiguousarray(kappa.T)

    for i in prange(N):
        for n in range(M):
            dW = random_numbers[i, n] * sqrt_delta_t
            Q_dot = np.dot(kappa_T, Q_traj[i, n, :]) - (1 / (2 * Wi)) * F(Q_traj[i, n, :])
            Q_traj[i, n + 1, :] = Q_traj[i, n, :] + Q_dot * delta_t + sqrt_1_Wi * dW

    return Q_traj

@njit
def compute_dyadic_product(Q, F_Q):
    return np.outer(Q, F_Q)

def compute_polymer_stress_tensor():
    np.random.seed(42)
    Q0 = np.random.multivariate_normal(Q0_mean, Q0_cov, N)
    Q0 = np.ascontiguousarray(Q0)
    n_steps = int(T / delta_t)
    random_numbers = np.random.normal(size=(N, n_steps, 2))
    random_numbers = np.ascontiguousarray(random_numbers)
    Q_traj_samples = euler_maruyama_vectorized(Q0, kappa, Wi, delta_t, T, random_numbers)
    final_states = Q_traj_samples[:, -1, :]

    sum_dyadic_product = np.zeros((2, 2))

    for i in range(N):
        Q = final_states[i]
        F_Q = F(Q)
        dyadic_product = compute_dyadic_product(Q, F_Q)
        sum_dyadic_product += dyadic_product

    E_dyadic_product = sum_dyadic_product / N
    tau_np = C * E_dyadic_product
    return tau_np

if __name__ == "__main__":
    tau_np = compute_polymer_stress_tensor()
    print("Polymer Stress Tensor:\n", tau_np)
