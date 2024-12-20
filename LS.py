import numpy as np
from scipy.linalg import solve
import matplotlib.pyplot as plt

def construct_basis_matrix_segmented(x, z):

    N = len(x)
    M = len(z) - 1
    B = np.zeros((N, M + 1))

    for i in range(N):
        for k in range(M + 1):
            if k > 0 and z[k - 1] <= x[i] <= z[k]:

                B[i, k] = (x[i] - z[k - 1]) / (z[k] - z[k - 1])
            if k < M and z[k] <= x[i] <= z[k + 1]:

                B[i, k] += (z[k + 1] - x[i]) / (z[k + 1] - z[k])
    return B

def least_squares_fit(x, f, M):

    z = np.linspace(min(x), max(x), M + 1)

    B = construct_basis_matrix_segmented(x, z)

    # минимизации ||Bu - f||^2
    BTB = B.T @ B
    BTf = B.T @ f
    u = solve(BTB, BTf)

    return u, B, z

def evaluate_approximation(u, B):

    return B @ u

def plot_results(x, f, x_dense, f_approx, z):

    plt.figure(figsize=(10, 6))
    plt.scatter(x, f, label="Исходные точки", color="red")
    plt.plot(x_dense, f_approx, label="Аппроксимация", color="blue")
    plt.vlines(z, min(f) - 1, max(f) + 1, colors="gray", linestyles="dashed", label="Контрольные точки")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Метод наименьших квадратов с сегментацией")
    plt.grid()
    plt.show()


def max_interpolation_error(f_sym, f_derivative_2, x_points, y_points):

    mid_points = (x_points[:-1] + x_points[1:]) / 2

    second_derivative_values = np.abs(f_derivative_2(mid_points))

    # Максимальная ошибка
    h = x_points[1] - x_points[0]
    return (h ** 2) / 8 * max(second_derivative_values)


x = np.linspace(0, 10, 15)
f = np.sin(x) + 0.2 * np.random.randn(len(x))


M = 5
u, B, z = least_squares_fit(x, f, M)

x_dense = np.linspace(min(x), max(x), 500)
B_dense = construct_basis_matrix_segmented(x_dense, z)
f_approx = evaluate_approximation(u, B_dense)



a, b = 0, np.pi
f_sym = lambda x: np.cos(x)
f_derivative_2 = lambda x: -np.cos(x)

N_values = [10, 20, 40, 80, 200]
results = []

for N in N_values:
    x_points = np.linspace(a, b, N + 1)
    y_points = f_sym(x_points)

    h = (b - a) / N
    max_error = max_interpolation_error(f_sym, f_derivative_2, x_points, y_points)
    c = max_error / (h ** 2)

    print(f"N = {N}, h = {h:.6f}, Max Error = {max_error:.6f}, C = {c:.6f}")


for N, max_error, c in results:
    print(f"N = {N}, Max Error = {max_error:.6f}, C = {c:.6f}")


plot_results(x, f, x_dense, f_approx, z)