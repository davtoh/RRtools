# http://docs.sympy.org/latest/modules/tensor/indexed.html
# http://docs.sympy.org/dev/_modules/sympy/concrete/summations.html
# http://docs.sympy.org/latest/modules/mpmath/calculus/sums_limits.html
# http://stackoverflow.com/questions/20886891/sympy-display-full-equation-in-dense-matrix-form
# http://mathworld.wolfram.com/PolygonArea.html
from preamble import *
from sympy import Sum, symbols, Subs, Abs, simplify, Eq, \
    solve, Symbol, IndexedBase, Indexed, Idx, latex, Matrix
from printer import libreOffice, getSteps, getEqs, printableSteps
from util import evalSum
import numpy as np

n, m, i,j = symbols('n m i j', integer=True)
x,y = IndexedBase('x', shape=(n,)),IndexedBase('y', shape=(n,))
#i,j = symbols('i j', cls=Idx)
#i,j = Idx('i', m),Idx('j', n)
# Indexed('x',symbols("i")) == x[i]

def shiftedMul(c1, c2, N, index, shift=0):
    """
    Expression to multiply two arrays and sum its elements: one static array with the other shifted.

    :param c1: (Array  1xN or Nx1 dim) static array.
    :param c2: (Array 1xN or Nx1 dim) shifted array.
    :param N: (positive symbol or int) number of coordinates in the any array).
    :param index: (positive symbol or int) index of the arrays.
    :param shift: (positive symbol or int) positional shifting.
    :return: expression.

    .. note:
        - Notice that both arrays should have the same dimensions.
        - There is no order for first and second array i.e polyArea_SYM(c1,c2) gives the same as polyArea_SYM(c1,c2).

    Example::

        i,n,s = symbols('i n s', integer=True)
        x,y = IndexedBase('x', shape=(n,)),IndexedBase('y', shape=(n,))
        mul_shit0 = shiftedMul(x, y, n, i, shift=0)
        print mul_shit0, " = ", mul_shit0.doit()
        # shift the same length
        mul_shit2 = shiftedMul(x, y, 2, i, shift=2)
        print mul_shit2, " = ", mul_shit2.doit()
        # shift
        mul_shit2 = shiftedMul(x, y, 4, i, shift=2)
        print mul_shit2, " = ", mul_shit2.doit()

    .. seealso: :func:`polyArea_SYM`
    """
    #N = abs(N) # dimensions are always positive
    shift = abs(shift) # necessary to not break the equation
    part1 = Sum(c1[index] * c2[index + shift], (index, 0, N - shift - 1)) # until shifted array ends
    part2 = Sum(c1[N - shift + index] * c2[index], (index, 0, shift - 1)) # continue were left off
    return part1 + part2

def polyArea_SYM(c1, c2, N=n, index=i):
    """
    Expression to find the area of polygon.

    :param c1: (Array 1xN or Nx1 dim) of x or y coordinates.
    :param c2: (Array 1xN or Nx1 dim) of y or x coordinates.
    :param N: (positive symbol or int) number of coordinates in the any array).
    :param index: (positive symbol or int) index of the arrays.
    :return: expression.

    .. note:
        - Notice that both arrays should have the same dimensions.
        - last points must be first point to complete polygon
        - There is no order for first and second array i.e polyArea_SYM(c1,c2) gives the same as polyArea_SYM(c1,c2).
        - :func:`shiftedMul` is part of polyArea_SYM and it is used to find the shifted multiplication.
        - If the polygon crosses over itself the algorithm will fail.
        - Based on http://www.mathopenref.com/coordpolygonarea.html

    .. warning: As a limitation, crossed polygons gives the wrong area.

    Example::

        import numpy as np
        ans = polyArea_SYM(x, y)
        points = np.array([[-3,-2],[-1,4],[6,1],[3,10],[-4,9],[-3,-2]]) # expected area 60
        xn, yn = points[:, 0], points[:, 1]
        f = evalSum((x,y,n),ans)
        print "Expanded equation: ",evalSum((n,),ans)(len(xn)-1)
        ans_num = f(xn,yn,len(xn)-1)
        assert 60==ans_num
        print "Test area passed): ", ans_num

    Proves that c1 and c2 are interchangeable::

        import numpy as np
        points = np.array([[-3,-2],[-1,4],[6,1],[3,10],[-4,9],[-3,-2]]) # expected area 60
        xn, yn = points[:, 0], points[:, 1]
        ans1 = polyArea_SYM(x,y) # build 1
        ans2 = polyArea_SYM(y,x) # build 2
        f1 = evalSum((x,y,n),ans1) # function of build 1
        f2 = evalSum((x,y,n),ans2) # function of build 2
        # test builds and order of xn and xy combinations
        assert f1(xn,yn,len(xn)-1)== f1(yn,xn,len(xn)-1)==f2(xn,yn,len(xn)-1)== f2(yn,xn,len(xn)-1)

    .. seealso: :func:`shiftedMul`
    """
    summatory = Sum(c1[index] * c2[index + 1] - c2[index] * c1[index + 1], (index, 0, N - 1))# simplified shidtedMul
    return Abs(summatory + c1[N] * c2[0] - c2[N] * c1[0]) / 2


def polyArea_STEPS(c1=x, c2=y, N=n, index=i, expressions ="expr,Xvar,Yvar"):
    """
    Gives the standard calculation steps to prove :func:`polyArea_SYM`.

    :param c1: (Array 1xN or Nx1 dim) of x or y coordinates.
    :param c2: (Array 1xN or Nx1 dim) of y or x coordinates.
    :param N: (positive symbol or int) number of coordinates in the any array).
    :param index: (positive symbol or int) index of the arrays.
    :param expressions: string of the three substituting expressions
    :return: steps i.e. list of ('symbol', 'replacing expression', 'result') items.

    Example::
        steps = polyArea_STEPS(x,y) # get steps
        lines = printableSteps(steps, converter=libreOffice) # process in readable form
        print "\n".join(lines) # get string print to standard output
        assert Eq(steps[-1][-1] - polyArea_SYM(x, y),0) == True # tests that last result is polyArea_SYM

    .. seealso:: :func:`getSteps`, :func:`polyAreas_LINES`, :func:`polyArea_SYM`
    """
    # http://www.wikihow.com/Calculate-the-Area-of-a-Polygon
    # http://www.mathopenref.com/coordpolygonarea.html
    if isinstance(expressions,basestring):
        expr,Xexpr,Yexpr = symbols(expressions)
    else:
        expr,Xexpr,Yexpr = expressions
    replace = {expr :Abs(Xexpr-Yexpr)/2,
               Xexpr:Sum(c1[index] * c2[index + 1], (index, 0, N - 1)) + c1[N] * c2[0],
               Yexpr:Sum(c2[index] * c1[index + 1], (index, 0, N - 1)) + c2[N] * c1[0]}
    steps = getSteps(expr, replace)
    steps.append((expr, None, polyArea_SYM(c1, c2)))
    return steps # answer

def polyAreas_LINES(c1=x, c2=y, N=n, index=i, converter=latex):
    """
    Gives the standard readable lines to prove :func:`polyArea_SYM`.

    :param c1: (Array 1xN or Nx1 dim) of x or y coordinates.
    :param c2: (Array 1xN or Nx1 dim) of y or x coordinates.
    :param N: (positive symbol or int) number of coordinates in the any array).
    :param index: (positive symbol or int) index of the arrays.
    :param converter: converter engine. default is latex
    :return: steps i.e. list of ('symbol', 'replacing expression', 'result') items.

    Example::
        lines = polyAreas_LINES(steps, converter=libreOffice) # process in readable form
        print "\n".join(lines) # get string print to standard output

    .. seealso:: :func:`getSteps`, :func:`printableSteps`, :func:`polyArea_STEPS`, :func:`polyArea_SYM`
    """
    steps = polyArea_STEPS(x, y) # get steps
    return printableSteps(steps, converter=converter) # process in readable form

if __name__ == "__main__":
    Pexpr,Xexpr,Yexpr = symbols("poligonArea,xsum,ysum")
    steps = polyArea_STEPS(x, y, expressions = (Pexpr,Xexpr,Yexpr))

    lines = printableSteps(steps,explanations=False,shroudwith="")

    ans = steps[-1][-1]
    assert Eq(ans - polyArea_SYM(x, y), 0) == True
    pts = [[-3,-2],[-1,4],[6,1],[3,10],[-4,9],[-3,-2]]
    lines.append(latex(Matrix(pts), mat_str = "array"))
    points = np.array(pts) # expected area 60
    xn, yn = points[:, 0], points[:, 1]
    f = evalSum((x,y,n),ans)
    ans_num = f(xn,yn,len(xn)-1)
    assert 60==ans_num
    pts2 = evalSum((n,),ans)(len(xn)-1)
    lines.append("{} = ".format(ans_num) + latex(pts2))
    #print("\n".join(lines))
