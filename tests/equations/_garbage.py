# http://docs.sympy.org/dev/modules/printing.html
# http://docs.sympy.org/latest/tutorial/printing.html
# http://docs.sympy.org/latest/tutorial/matrices.html
# http://docs.sympy.org/0.7.2/modules/matrices/expressions.html
# http://web.cs.iastate.edu/~cs577/handouts/homogeneous-transform.pdf

from sympy import symbols, Matrix, MatrixSymbol, Eq, S, collect, summation, Subs, Sum, Indexed
from sympy.functions import sin,cos
from libreOffice import print_libreOffice
from sympy.printing import print_latex

def HXtranslate(x):
    return Matrix([[1,0,x],[0,1,0],[0,0,1]])

def HYtranslate(y):
    return Matrix([[1,0,0],[0,1,y],[0,0,1]])

def HZtranslate(z):
    return Matrix([[1,0,0],[0,1,0],[0,0,z]])

def Htranslate():
    return Matrix([[1,0,0],[0,1,0],[0,0,1]])

def HXrotate(theta):
    """
    :param theta: in radians
    :return: H
    """
    return Matrix([[1,0,0],[0,cos(theta),-sin(theta)],[0,sin(theta),cos(theta)]])

def HYrotate(theta):
    """
    :param theta: in radians
    :return: H
    """
    return Matrix([[cos(theta),0,sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]])

def HZrotate(theta):
    """
    :param theta: in radians
    :return: H
    """
    return Matrix([[cos(theta),-sin(theta),0],[sin(theta),cos(theta),0],[0,0,1]])

def Hrotate():
    """
    :param theta: in radians
    :return: H
    """
    return Matrix([[1,0,0],[0,1,0],[0,0,1]])

def HXscale(x):
    return Matrix([[x,0,0],[0,1,0],[0,0,1]])

def HYscale(y):
    return Matrix([[1,0,0],[0,y,0],[0,0,1]])

def HZscale(z):
    return Matrix([[1,0,0],[0,1,0],[0,0,z]])

def Hscale(x=1, y=1, z = 1):
    return Matrix([[x,0,0],[0,y,0],[0,0,z]])

def polyArea():
    from sympy.tensor import IndexedBase, Idx
    from sympy import symbols
    x,y = IndexedBase('x'),IndexedBase('y')
    n, m, i,j = symbols('n m i j', integer=True)
    #i,j = symbols('i j', cls=Idx)
    #i,j = Idx('i', m),Idx('j', n)
    Xexpr = Sum(x[i]*y[i+1]+x[m]*y[0],(i,0,m-1))
    Yexpr = Sum(y[j]*x[j+1]+y[m]*x[0],(j,0,n-1))
    print_latex(Xexpr)
    print_latex(Yexpr)
    pass

def EXAMPLE():
    i = symbols("i")
    s=Sum(Indexed('x',i),(i,1,3))
    def ss(x):
        r = range(s.limits[0][1], s.limits[0][2] + 1)
        l = [s.function.subs(s.variables[0], j) for j in r]
        return Subs(s.doit(), l, x).doit()
    ss((1,2,3))

Aow,Aoh = symbols("W_oA,H_oA")
Asw,Ash = symbols("W_sA,H_sA")
Bow,Boh = symbols("W_oB,H_oB")
Bsw,Bsh = symbols("W_sB,H_sB")
oAToB,oATsA,sBToB = symbols("oAToB,oATsA,sBToB")

_oATsA = Matrix([[Asw/Aow,0,0],[0,Ash/Aoh,0],[0,0,1]])
sATsB = MatrixSymbol("M",3,3)
_sATsB = Matrix(sATsB)
_sBToB = Matrix([[Bow/Bsw,0,0],[0,Boh/Bsh,0],[0,0,1]])
#print_latex(Eq(oAToB, sBToB*sATsB*oATsA))
#print_latex(Eq(oAToB, _sBToB*sATsB*oATsA_mat))
#print_latex((_sBToB*sTM_mat*oATsA_mat))
polyArea()