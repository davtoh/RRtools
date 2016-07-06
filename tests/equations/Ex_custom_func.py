# http://docs.sympy.org/dev/modules/utilities/autowrap.html
# https://ojensen.wordpress.com/
# http://www.programcreek.com/python/example/38815/sympy.lambdify
from sympy import Sum, symbols
from sympy.utilities.lambdify import lambdify, implemented_function
from sympy.tensor import IndexedBase, Idx
n, m, i, j = symbols('n m i j x y', integer=True)
x, y = symbols('x y')
A = IndexedBase('A', shape=(n,))
print A.shape

def SumH(x):
    return 'sin(%s) is cool' % x
myfuncs = {"Sum" : SumH}
f = lambdify(x, sin(x), myfuncs); f(1)
'sin(1) is cool'
"""
test_f = lambdify((x,y), x*y)
print test_f(2,10)"""