import numpy as np
from scipy.integrate import quad



g = 9.81
alpha = np.sqrt(2 * g)

def f1(x):
    return x
def df1(x):
    return 1

def f2(x):
    return x * x

def df2(x):
    return 2 * x



def integrand(x, f, df):
    return np.sqrt((1 + df(x) ** 2) / (1 - f(x)))



def composite_simpson_rule(x, f):

    h = x[1] - x[0]  # Шаг
    S = 0
    for i in range(len(x) - 1):

        S += h / 6 * (f(x[i]) + f(x[i + 1]) + 4 * f((x[i] + x[i + 1]) / 2))
    return S


def composite_trapezoidal_rule(x, f):

    h = x[1] - x[0]  # Шаг
    S = 0
    for i in range(len(x) - 1):

        S += ((f(x[i]) + f(x[i + 1])) / 2) * h
    return S




def adaptive_integration(a, b, f, df, method, epsilon=1e-4):
    N = 2
    x = np.linspace(a, b, N + 1)
    S_prev = method(x, lambda x: integrand(x, f, df))

    while True:
        N *= 2
        x = np.linspace(a, b, N + 1)
        S = method(x, lambda x: integrand(x, f, df))
        if abs(S - S_prev) < epsilon:
            break
        S_prev = S
    h = (b - a) / N
    return S, h



def evaluate_integral_with_adaptive(f, df, method, epsilon=1e-4):
    a = 0
    b = 1 - 1e-6  # Исключаем деление на 0
    exact_integral, _ = quad(lambda x: integrand(x, f, df), a, b)
    T_exact = exact_integral / alpha


    approx_integral,h = adaptive_integration(a, b, f, df, method, epsilon)
    T_approx = approx_integral / alpha
    error = abs(T_exact - T_approx)
    c = error / h ** 2

    return T_exact, T_approx, error, c,h



def main():
    functions = [(f1, df1, "f(x) = x"), (f2, df2, "f(x) = x^2")]
    methods = [("trapezoidal", composite_trapezoidal_rule), ("simpson", composite_simpson_rule)]

    for f, df, label in functions:
        print(f"\nResults for {label}:")
        for method_name, method in methods:
            print(f"\nUsing {method_name} method:")
            T_exact, T_approx, error,c,h = evaluate_integral_with_adaptive(f, df, method)
            print(f"Exact T: {T_exact:.6f}")
            print(f"Approximate T: {T_approx:.6f}")
            print(f"Error: {error:.6f}")

            print(f"Constant h: {h:.6e}")


if __name__ == "__main__":
    main()
