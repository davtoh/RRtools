import os
path = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from sympy import symbols, Idx, IndexedBase, Eq, source
import inspect
from sympy.utilities.autowrap import autowrap
from sympy.utilities.lambdify import lambdify, implemented_function
a, b = symbols('a b')
m = symbols("m",integer = True)
i = Idx('i', m)
linearmap = a + (i - 1)*(a - b)/(1 - m)
x = IndexedBase('x')
eqs = Eq(x[i], linearmap)
linspace = autowrap(eqs,tempdir=path)
#linspace = lambdify((a,b,m),linearmap,'numpy')
print linspace(0, 1, 5)
