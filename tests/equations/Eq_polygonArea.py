# http://docs.sympy.org/latest/modules/tensor/indexed.html
# http://docs.sympy.org/dev/_modules/sympy/concrete/summations.html
# http://docs.sympy.org/latest/modules/mpmath/calculus/sums_limits.html
# http://stackoverflow.com/questions/20886891/sympy-display-full-equation-in-dense-matrix-form
# http://mathworld.wolfram.com/PolygonArea.html
# http://stackoverflow.com/a/10138307/5288758
from preamble import *
from polygonArea import *

def eq(pts=None):
    Polyvar, Xvar, Yvar = symbols("poligonArea,xsum,ysum")
    steps = polyArea_STEPS(x, y, expressions = (Polyvar, Xvar, Yvar))
    Pexpr1 = steps[0][1] # Poly expression
    Yexpr1 = steps[1][1] # Y expression
    Xexpr1 = steps[2][1] # X expression
    Pexpr2 = steps[3][2] # Poly replaced expression

    assert Eq(Pexpr2 - polyArea_SYM(x, y), 0) == True

    if pts is None: # used if it is defined outside scope
        pts = [[-3,-2],[-1,4],[6,1],[3,10],[-4,9],[-3,-2]]

    points = np.array(pts) # expected area 60
    strpts = latex(Matrix(pts), mat_str = "array")
    xn, yn = points[:, 0], points[:, 1] # get x,y coordinates

    ans_Yexpr = Yexpr1.subs(n,len(xn)-1).doit() # Yexpr1 indexes
    ans_Xexpr = Xexpr1.subs(n,len(xn)-1).doit() # Xexpr1 indexes
    ans_Pexpr = evalSum((n,),Pexpr2)(len(xn)-1) # Pexpr2.subs(n,len(xn)-1).doit()

    variables = indexMap(x,xn)+indexMap(y,yn) # get mapping
    ans_Yexpr1 = replaceIndexes(ans_Yexpr,variables) # Yexpr1 replaced indexes (str)
    ans_Xexpr1 = replaceIndexes(ans_Xexpr,variables) # Xexpr1 replaced indexes (str)
    ans_Yexpr2 = parse_expr(ans_Yexpr1) # Yexpr1 replaced indexes (expr)
    ans_Xexpr2 = parse_expr(ans_Xexpr1) # Xexpr1 replaced indexes (expr)

    ans_Yexpr1 = replaceIndexes(ans_Yexpr, variables + [("*"," \\cdot ")])
    ans_Xexpr1 = replaceIndexes(ans_Xexpr, variables + [("*"," \\cdot ")])
    ans_Pexpr1 = replaceIndexes(Pexpr1,[(Xvar,ans_Xexpr2),(Yvar,ans_Yexpr2)]) # Pexpr1 repleced
    ans_Pexpr2 = Pexpr1.subs({Xvar:ans_Xexpr2,Yvar:ans_Yexpr2}) # answer
    return locals()

if __name__ == "__main__":
    globals().update(eq())