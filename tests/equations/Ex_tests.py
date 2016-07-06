
from sympy import Sum, symbols, Subs, Abs, simplify, Eq, solve, Symbol, IndexedBase, Indexed, Idx, latex
from printer import libreOffice, getSteps, getEqs, printableSteps
from util import evalSum

n, m, i,j = symbols('n m i j', integer=True)
x,y = IndexedBase('x', shape=(n,)),IndexedBase('y', shape=(n,))

def test():
    """
    Test to see how evaluation of expressions work
    :return:
    """
    #i = symbols("i")
    s=Sum(x[i]+i,(i,1,3))#Sum(Indexed('x',i),(i,1,3))
    s2=Sum(x[i]+y[i],(i,n,m))
    def ss(x):
        r = range(s.limits[0][1], s.limits[0][2] + 1)
        l = [list(s.function.subs(s.limits[0][0], j).atoms(Indexed))[0] for j in r]
        return s.doit().subs(dict(zip(l,x)))#Subs(s.doit(), l, x).doit()
    ss((2,3,5))
    evalSum((x,n,m),s2)((1,2,3),0,1)