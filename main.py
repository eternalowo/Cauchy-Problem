from math import e, sin, cos
import matplotlib.pyplot as plt


def correct_sollution(x):
    """Exact solution"""
    return e ** x + 4 * e ** (2 * x) - 0.2 * sin(x) - 0.6 * cos(x)


def func(x, y):
    """Cauchy problem (y'' = 2sin(x) + 3y' - 2y)"""
    return 2 * sin(x) + 3 * y[1] - 2 * y[0]


def calc_h(x0, xm, m):
    """Function for calculating the step based on the start point, end point and number of split segments"""
    return (xm - x0) / m


def euler_method(x, h, y_list_e):
    """Function for calculating the next value by the Euler method"""
    delta_y1 = h * y_list_e[-1][1]
    delta_y2 = h * func(x, y_list_e[-1])
    y1 = y_list_e[-1][0] + delta_y1
    y2 = y_list_e[-1][1] + delta_y2
    return y1, y2


def euler_cauchy_method(x, h, y_list_c):
    """Function for calculating the next value by the Euler-Cauchy method"""
    k11 = h * y_list_c[-1][1]
    k12 = h * func(x, y_list_c[-1])
    k21 = h * (y_list_c[-1][1] + k12)
    k22 = h * func(x + h, (y_list_c[-1][0] + k11, y_list_c[-1][1] + k12))

    delta_y1 = (k11 + k21) / 2
    delta_y2 = (k12 + k22) / 2

    y1 = y_list_c[-1][0] + delta_y1
    y2 = y_list_c[-1][1] + delta_y2

    return y1, y2


def runge_kutta_method(x, h, y_list_rk):
    """Function for calculating the next value by the Runge-Kutta method of the fourth order of accuracy"""
    k11 = h * y_list_rk[-1][1]
    k12 = h * func(x, y_list_rk[-1])
    k21 = h * (y_list_rk[-1][1] + k12 / 2)
    k22 = h * func(x + h / 2, (y_list_rk[-1][0] + k11 / 2, y_list_rk[-1][1] + k12 / 2))
    k31 = h * (y_list_rk[-1][1] + k22 / 2)
    k32 = h * func(x + h / 2, (y_list_rk[-1][0] + k21 / 2, y_list_rk[-1][1] + k22 / 2))
    k41 = h * (y_list_rk[-1][1] + k32)
    k42 = h * func(x + h, (y_list_rk[-1][0] + k31, y_list_rk[-1][1] + k32))

    delta_y1 = (k11 + 2 * k21 + 2 * k31 + k41) / 6
    delta_y2 = (k12 + 2 * k22 + 2 * k32 + k42) / 6

    y1 = y_list_e[-1][0] + delta_y1
    y2 = y_list_e[-1][1] + delta_y2

    return y1, y2


# Example

x_init = 0
x_end = 1
n = 100
y_0 = (2.6, 3.2)

# Calculating step
h = calc_h(x_init, x_end, n)

# Arrays for derivatives values
y_list_e = [y_0]
y_list_c = [y_0]
y_list_rk = [y_0]

# Calculating derivatives
for i in range(n):
    x_init += h
    y_list_e.append(euler_method(x_init, h, y_list_e))
    y_list_c.append(euler_cauchy_method(x_init, h, y_list_c))
    y_list_rk.append(runge_kutta_method(x_init, h, y_list_rk))

# Arrays to show results
x_res = []
res_euler = []
res_runge_kutta = []
res_cauchy_euler = []
res_exact_solution = []

x_init = 0
for i in range(n):
    x_init += h
    x_res.append(x_init)
    res_euler.append(func(x_init, y_list_e[i]))
    res_cauchy_euler.append(func(x_init, y_list_c[i]))
    res_runge_kutta.append(func(x_init, y_list_rk[i]))
    res_exact_solution.append(correct_sollution(x_init))
    # print("(", x_init, ",", func(x_init, y_list_e[i]), ")")
    # print("(", x_init, ",", func(x_init, y_list_c[i]), ")")
    # print("(", x_init, ",", func(x_init, y_list_rk[i]), ")")


# Plotting results
plt.plot(x_res, res_euler, linewidth=6, label='Euler')
plt.plot(x_res, res_cauchy_euler, linestyle='dashed', linewidth=5, label='Euler-Cauchy')
plt.plot(x_res, res_runge_kutta, linestyle=':', color='gold', linewidth=4, label='Runge-Kutta 4')
plt.plot(x_res, res_exact_solution, linestyle=':', color='blue', linewidth=4, label='y')
plt.xlabel('x')
plt.ylabel('y')
plt.title("y'' - 3y' + 2y = 2sin(x)")
plt.legend()
plt.show()
