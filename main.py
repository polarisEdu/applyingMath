import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


def piecewise_linear_interpolation(x_points, y_points, x):
    N = len(x_points) - 1
    for i in range(N):
        if x_points[i] <= x <= x_points[i + 1]:
            h_i = x_points[i + 1] - x_points[i]
            return y_points[i] * (x_points[i + 1] - x) / h_i + y_points[i + 1] * (x - x_points[i]) / h_i
    return None


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

        # Вторая производная в середине отрезка
        second_derivative_value = abs(f_second_derivative_func(midpoint))

        # Оценка ошибки на интервале
        error = (second_derivative_value / 2) * (h_i ** 2)
        max_error = max(max_error, error)

    return max_error



x = sp.Symbol('x')
f_sym = sp.cos(x)
f_derivative_2 = sp.diff(f_sym, x, 2)


a, b = 0, np.pi
N_values = [10, 20, 40, 80]

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

y_interp = [piecewise_linear_interpolation(x_points_plot, y_points_plot, x) for x in x_fine]


plt.figure(figsize=(10, 6))
plt.plot(x_fine, y_fine, label=r'$\cos(x)$', color='blue')
plt.plot(x_fine, y_interp, label='Piecewise Linear Interpolation', linestyle='--', color='orange')
plt.scatter(x_points_plot, y_points_plot, color='red', label='Interpolation Points')
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Piecewise Linear Interpolation vs. cos(x)')
plt.legend()
plt.grid(True)
plt.show()



# data_points = np.array([
#     [0, 0],
#     [50, 20],
#     [60, 27],
#     [70, 35],
#     [80, 43],
#     [90, 53],
#     [100, 63]
#
# ])

# x_points = data_points[:, 0]
# y_points = data_points[:, 1]
#

# x_fine = np.linspace(min(x_points), max(x_points), 500)
# y_fine = np.array([piecewise_linear_interpolation(x_points, y_points, x) for x in x_fine])
#

# plt.plot(x_points, y_points, 'bo', label="Данные (узлы интерполяции)")
# plt.plot(x_fine, y_fine, 'r-', label="Интерполированная функция")
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.legend()
# plt.grid(True)
# plt.show()