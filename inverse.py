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


def inverse_piecewise_linear_interpolation(x_points, y_points, y):
    """
    Вычисляет x = g_h(y), обратную функцию для кусочной линейной интерполяции.

    x_points: массив узлов x
    y_points: массив значений функции в узлах x
    y: значение, для которого нужно найти x

    Возвращает список решений x (может быть несколько, если y встречается на нескольких интервалах).
    """
    solutions = []
    N = len(x_points) - 1  # Количество интервалов

    for i in range(N):
        # Проверяем, лежит ли y в диапазоне текущего интервала
        if min(y_points[i], y_points[i + 1]) <= y <= max(y_points[i], y_points[i + 1]):
            # Вычисляем x по формуле обратной функции
            x = x_points[i] + (y - y_points[i]) * (x_points[i + 1] - x_points[i]) / (y_points[i + 1] - y_points[i])
            solutions.append(x)

    return solutions


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


# Прямая интерполяция cos(x)
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

# Построение графиков прямой интерполяции
N_plot = 10
x_points_plot = np.linspace(a, b, N_plot + 1)
y_points_plot = np.cos(x_points_plot)

x_fine = np.linspace(a, b, 500)
y_fine = np.cos(x_fine)

y_interp = [piecewise_linear_interpolation(x_points_plot, y_points_plot, x) for x in x_fine]

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
x_inverse = [inverse_piecewise_linear_interpolation(x_points_plot, y_points_plot, y)[0] for y in y_fine_inverse]

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

# Проверка обратной функции
y_values = [0.5, -0.5]
for y in y_values:
    x_solutions = inverse_piecewise_linear_interpolation(x_points_plot, y_points_plot, y)
    print(f"For y = {y}, x solutions = {x_solutions}")
