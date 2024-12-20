import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def spline_coefficients(x, f):
    x = np.array(x, dtype=np.float64)
    f = np.array(f, dtype=np.float64)
    n = len(x) - 1
    h = np.diff(x)

    # Правая часть системы
    F = np.zeros(n - 1, dtype=np.float64)
    for i in range(1, n):
        F[i - 1] = (f[i + 1] - f[i]) / h[i] - (f[i] - f[i - 1]) / h[i - 1]

    # Сборка матрицы
    diagonals = [
        h[:-1],
        2 * (h[:-1] + h[1:]),
        h[1:]
    ]
    A = diags(diagonals, offsets=[-1, 0, 1], shape=(n - 1, n - 1), dtype=np.float64)

    # # Регуляризация
    # A = A + diags([1e-15 * np.ones(n - 1, dtype=np.float64)], [0])

    # Преобразование в формат CSR
    A = csr_matrix(A)

    # Решение системы
    y = spsolve(A, 6 * F)

    # Коэффициенты
    c = np.zeros(n + 1, dtype=np.float64)
    c[1:n] = y
    c[0] = c[-1] = 0  # Натуральные условия

    a = f[:-1]
    b = np.diff(f) / h - h * (2 * c[:-1] + c[1:]) / 6
    d = np.diff(c) / h

    return a, b, c, d, h




def spline_evaluate(a, b, c, d, x, h, x_eval):
    spline_values = np.zeros_like(x_eval)

    for i in range(len(x) - 1):
        mask = (x_eval >= x[i]) & (x_eval <= x[i + 1])
        dx = x_eval[mask] - x[i]
        spline_values[mask] = (
            a[i]
            + b[i] * dx
            + c[i] * dx**2 / 2
            + d[i] * dx**3 / 6
        )

    return spline_values


# def spline_derivative(a, b, c, d, x, h, x_eval):
#
#     derivative_values = np.zeros_like(x_eval)
#     for i in range(len(x) - 1):
#         mask = (x_eval >= x[i]) & (x_eval <= x[i + 1])
#         dx = x_eval[mask] - x[i]
#         derivative_values[mask] = (
#             b[i] + c[i] * dx + d[i] * dx**2 / 2
#         )
#     return derivative_values

def find_extrema(a, b, c, d, x, h):

    x_extrema = []
    for i in range(len(x) - 1):

        if abs(d[i]) > 1e-12:
            roots = np.roots([d[i] / 2, c[i], b[i]])
            for root in roots:
                if 0 <= root <= h[i]:
                    x_extrema.append(x[i] + root)
        elif abs(c[i]) > 1e-12:
            root = -b[i] / c[i]
            if 0 <= root <= h[i]:
                x_extrema.append(x[i] + root)
    return np.sort(x_extrema)

def find_all_roots(f_spline, x, y0, tol=1e-8, epsilon=1e-6):

    a, b, c, d, h = spline_coefficients(x, f)
    x_extrema = find_extrema(a, b, c, d, x, h)


    x_check = np.sort(np.unique(np.concatenate((x, x_extrema))))

    roots = []
    for i in range(len(x_check) - 1):
        x_left, x_right = x_check[i], x_check[i + 1]
        f_left = f_spline(x_left)
        f_right = f_spline(x_right)


        if (f_left - y0) * (f_right - y0) <= 0 or \
           abs(f_left - y0) < epsilon or abs(f_right - y0) < epsilon:
            root = secant_method_inverse(f_spline, x, y0, x_left, x_right, tol=tol)
            if x_left <= root <= x_right:
                roots.append(root)
    return roots




def secant_method_inverse(f_spline, x, y0, x0, x1, tol=1e-6, max_iter=100):

    for _ in range(max_iter):
        f0 = f_spline(x0) - y0
        f1 = f_spline(x1) - y0

        if abs(f1 - f0) < tol:
            return x1

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        x0, x1 = x1, x2

        if abs(x1 - x0) < tol:
            return x1

    raise ValueError("Метод секущих не сошелся за максимальное число итераций.")

def cubic_spline(x,f, x_eval):
    a, b, c, d, h = spline_coefficients(x, f)
    return spline_evaluate(a, b, c, d, x, h, x_eval)


def inverse_spline_interpolation(x, f, y_eval):

    a, b, c, d, h = spline_coefficients(x, f)
    x_inverse = []
    for y in y_eval:
        roots = find_all_roots(f_spline, x, y)
        x_inverse.append(roots)
    return x_inverse





def f_spline(x_eval):
    a, b, c, d, h = spline_coefficients(x, f)
    return spline_evaluate(a, b, c, d, x, h, np.array([x_eval]))[0]

x = np.linspace(0, 10, 11)
f = np.sin(x)

y_eval = np.linspace(-1, 1, 500)
x_eval = np.linspace(x[0], x[-1], 500)
spline_values = cubic_spline(x,f, x_eval)
x_inverse = inverse_spline_interpolation(x, f, y_eval)
#print(x_inverse)
#print(y_eval)


# Пример: найти x, при котором S(x) = 0.5
y0 = 0.5

# Найти все корни, где S(x) = 0.5
roots = find_all_roots(f_spline, x, y0)

print(f"Все найденные корни: {roots}")




node_counts = [50,400,900]
errors = []
constants = []


print(f"{'Количество узлов':<20} {'Максимальная ошибка':<25} {'h_max':<20} {'Константа C':<20}")
for n in node_counts:
    x_nodes = np.linspace(0, 10, n)
    f_nodes = np.sin(x_nodes)

    a, b, c, d, h = spline_coefficients(x_nodes, f_nodes)
    h_max = np.max(h)

    x_test = np.linspace(0, 10, 500)
    real_error = np.abs(np.sin(x_test) - cubic_spline(x_nodes, f_nodes, x_test))

    max_error = np.average(real_error)


    C = max_error / (h_max ** 4)

    errors.append(max_error)
    constants.append(C)

    print(f"{n:<20} {max_error:<25.6e} {h_max:<20.6e} {C:<20.6e}")


plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.plot(x_eval, spline_values, label="Cubic Spline", color='blue')
plt.scatter(x, f, color='red', label="Data Points")
plt.axhline(y0, color="blue", linestyle="--", label=f"y = {y0}")
plt.title("Прямая интерполяция")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()


plt.subplot(1, 2, 2)
for i, y in enumerate(y_eval):
    roots = x_inverse[i]
    for root in roots:
        plt.scatter(y, root, color="green", label="Inverse Interpolation Points" if i == 0 else "")


plt.title("Обратная интерполяция")
plt.xlabel("y")
plt.ylabel("x")
plt.legend()
plt.grid()

plt.show()