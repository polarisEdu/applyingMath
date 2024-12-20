import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def piecewise_linear_interpolation_vectorized(x_points, y_points, x_values):

    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)
    x_values = np.asarray(x_values)


    indices = np.searchsorted(x_points, x_values, side='right') - 1
    indices = np.clip(indices, 0, len(x_points) - 2)


    h_i = x_points[indices + 1] - x_points[indices]
    y_interp = y_points[indices] * (x_points[indices + 1] - x_values) / h_i + \
               y_points[indices + 1] * (x_values - x_points[indices]) / h_i
    return y_interp


def inverse_piecewise_linear_interpolation_vectorized(x_points, y_points, y_values):

    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)
    y_values = np.asarray(y_values)


    if not np.all(y_points[:-1] <= y_points[1:]):
        sort_indices = np.argsort(y_points)
        y_points = y_points[sort_indices]
        x_points = x_points[sort_indices]


    indices = np.searchsorted(y_points, y_values, side='right') - 1
    indices = np.clip(indices, 0, len(y_points) - 2)


    x_inverse = x_points[indices] + (y_values - y_points[indices]) * \
                (x_points[indices + 1] - x_points[indices]) / \
                (y_points[indices + 1] - y_points[indices])
    return x_inverse



def max_interpolation_error(f, f_second_derivative, x_points, y_points):
    max_error = 0
    N = len(x_points) - 1
    x = sp.Symbol('x')

    # Преобразуем вторую производную в функцию
    f_second_derivative_func = sp.lambdify(x, f_second_derivative, modules=['numpy'])

    for i in range(N):
        # Находим h_i и промежуточную точку
        h_i = x_points[i + 1] - x_points[i]
        midpoint = (x_points[i] + x_points[i + 1]) / 2

        second_derivative_value = abs(f_second_derivative_func(midpoint))

        error = (second_derivative_value / 2) * (h_i ** 2)
        max_error = max(max_error, error)

    return max_error



x = sp.Symbol('x')
f_sym = sp.cos(x)
f_derivative_2 = sp.diff(f_sym, x, 2)

a, b = 0, np.pi
N_values = [10, 20, 40, 80, 200]

results = []

for N in N_values:
    x_points = np.linspace(a, b, N + 1)
    y_points = np.cos(x_points)

    # Вычисляем ||f - f_h|| / h^2
    h = (b - a) / N
    max_error = max_interpolation_error(f_sym, f_derivative_2, x_points, y_points)
    c = max_error / (h ** 2)
    results.append((N, max_error, c))

print(f"{'N':<5} {'Max Error':<15} {'C Value':<10}")
for N, max_error, c in results:
    print(f"{N:<5} {max_error:<15.8f} {c:<10.8f}")


N_plot = 10
x_points_plot = np.linspace(a, b, N_plot + 1)
y_points_plot = np.cos(x_points_plot)

x_fine = np.linspace(a, b, 500)
y_fine = np.cos(x_fine)


y_interp = piecewise_linear_interpolation_vectorized(x_points_plot, y_points_plot, x_fine)

plt.figure(figsize=(12, 6))

# График f_h(x)
plt.subplot(1, 2, 1)
plt.plot(x_fine, y_fine, label=r'$\cos(x)$', color='blue')
plt.plot(x_fine, y_interp, label='Piecewise Linear Interpolation', linestyle='--', color='orange')
plt.scatter(x_points_plot, y_points_plot, color='red', label='Interpolation Points')
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Piecewise Linear Interpolation f_h(x)')
plt.legend()
plt.grid(True)

# Построение обратной функции g_h(y)
y_fine_inverse = np.linspace(min(y_points_plot), max(y_points_plot), 500)
x_inverse = inverse_piecewise_linear_interpolation_vectorized(x_points_plot, y_points_plot, y_fine_inverse)

# График g_h(y)
plt.subplot(1, 2, 2)
plt.plot(y_fine_inverse, x_inverse, label='Inverse Function g_h(y)', color='green')
plt.scatter(y_points_plot, x_points_plot, color='orange', label='Interpolation Points (Reversed)')
plt.xlabel('y', fontsize=12)
plt.ylabel('x', fontsize=12)
plt.title('Inverse Function g_h(y)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# y_values = [0.5, -0.5]
# x_solutions = inverse_piecewise_linear_interpolation_vectorized(x_points_plot, y_points_plot, y_values)
# for y, x_sol in zip(y_values, x_solutions):
#     print(f"For y = {y}, x solution = {x_sol}")



# data_points = np.array([
#     [0, 0],
#     [50, 20],
#     [60, 27],
#     [70, 35],
#     [80, 43],
#     [90, 53],
#     [100, 63]
# ])
#
# x_points = data_points[:, 0]
# y_points = data_points[:, 1]
#
# x_fine = np.linspace(x_points[0], x_points[-1], 500)
#
#
# y_interp = piecewise_linear_interpolation_vectorized(x_points, y_points, x_fine)
#
# y_fine_inverse = np.linspace(y_points[0], y_points[-1], 500)
#
# x_inverse = inverse_piecewise_linear_interpolation_vectorized(x_points, y_points, y_fine_inverse)
# plt.figure(figsize=(12, 6))
#
# plt.subplot(1, 2, 1)
# plt.plot(x_fine, y_interp, label='Piecewise Linear Interpolation', linestyle='--', color='orange')
# plt.scatter(x_points, y_points, color='red', label='Data Points')
# plt.xlabel('x', fontsize=12)
# plt.ylabel('y', fontsize=12)
# plt.title('Piecewise Linear Interpolation')
# plt.legend()
# plt.grid(True)
#
#
# plt.subplot(1, 2, 2)
# plt.plot(y_fine_inverse, x_inverse, label='Inverse Function', color='green')
# plt.scatter(y_points, x_points, color='orange', label='Data Points (Reversed)')
# plt.xlabel('y', fontsize=12)
# plt.ylabel('x', fontsize=12)
# plt.title('Inverse Function')
# plt.legend()
# plt.grid(True)
#
# plt.tight_layout()
# plt.show()