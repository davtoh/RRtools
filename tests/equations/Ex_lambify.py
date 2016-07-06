# https://ocefpaf.github.io/python4oceanographers/blog/2014/01/27/batman/
# http://www.programcreek.com/python/example/38815/sympy.lambdify
from sympy import Function, symbols, sqrt, S

x, y = symbols("x y")
red = S("(x / 7) ** 2 * sqrt(abs(abs(x) - 3) / (abs(x) - 3)) + (y / 3) ** 2 * sqrt(abs(y + 3 / 7 * sqrt(33)) / (y + 3 / 7 * sqrt(33))) - 1")
orange = S("abs(x / 2) - ((3 * sqrt(33) - 7) / 112) * x ** 2 - 3 + sqrt(1 - (abs(abs(x) - 2) - 1) ** 2) - y")
green = S("9 * sqrt(abs((abs(x) - 1) * (abs(x) - 0.75)) / ((1 - abs(x)) * (abs(x) - 0.75))) - 8 * abs(x) - y")
blue = S("3 * abs(x) + 0.75 * sqrt(abs((abs(x) - 0.75) * (abs(x) - 0.5)) / ((0.75 - abs(x)) * (abs(x) - 0.5))) - y")
pink = S("2.25 * sqrt(abs((x - 0.5) * (x + 0.5)) / ((0.5 - x) * (0.5 + x))) - y")
brown = S("6 * sqrt(10) / 7 + (1.5 - 0.5 * abs(x)) * sqrt(abs(abs(x) - 1) / (abs(x) - 1)) - (6 * sqrt(10) / 14) * sqrt(4 - (abs(x) - 1) ** 2) - y")

from sympy import lambdify

def lamb(func):
    return lambdify((x, y), func, modules='numpy')

red, orange, green, blue, pink, brown = map(lamb, (red, orange, green, blue, pink, brown))


import numpy as np

spacing = 0.01
x, y = np.meshgrid(np.arange(-7.25, 7.25, spacing),
                   np.arange(-5, 5, spacing))

red = red(x, y)
orange = orange(x, y)
green = green(x, y)
blue = blue(x, y)
pink = pink(x, y)
brown = brown(x, y)


import matplotlib.pyplot as plt

colors = dict(red='#FF0000', orange='#FFA500', green='#008000',
              blue='#003399', pink='#CC0033', brown='#800000')

equations = dict(red=red, orange=orange, green=green,
                 blue=blue, pink=pink, brown=brown)

fig, ax = plt.subplots(figsize=(8, 6))
for key, color in colors.items():
    ax.contour(x, y, equations[key], [0], colors=color, linewidths=3)

_ = ax.set_xticks([])
_ = ax.set_yticks([])

plt.show()