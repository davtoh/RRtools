# http://docs.sympy.org/dev/tutorial/manipulation.html
# http://docs.sympy.org/dev/guide.html
## IMPORTANT http://docs.sympy.org/dev/modules/printing.html#module-sympy.printing.mathml
# http://tex.stackexchange.com/a/80351/97194
# http://docs.sympy.org/dev/modules/core.html

from sympy.printing.codeprinter import Assignment
from sympy import Add, Mul, Pow, Rational, Integer
from sympy import preorder_traversal, postorder_traversal,Matrix, MatrixSymbol, MatrixBase, Eq, S, collect, symbols, Symbol
from sympy.matrices.expressions.matexpr import MatrixExpr,MatrixSymbol,MatrixElement,MatMul,MatAdd,MatPow,Transpose,Inverse
from sympy.functions import sin,cos
from sympy.printing import print_latex, latex
from libreOffice import libreOffice, print_libreOffice
#from sympy.core.sympify import kernS

def getEqs(*args, **subs):
    """
    Get the equations.

    :param expr: the initial expression
    :param **subs: dictionary of substitutions as the form 'symbol' = 'replacing expression'
    :return:
    """
    assert len(args)==1, "getEqs only accepts one expression"
    expr = args[0] # get first expression
    equations = [] # initialize equations
    while True:
        newexp = expr.subs(subs)
        if expr != newexp:
            equations.append(Eq(expr,newexp))
            expr = newexp
        else:
            break
    equations.append(Eq(args[0],expr))
    return equations

def getSteps(expr, subs):
    """
    Get substitution steps.

    :param expr: the initial expression
    :param subs: dictionary of substitutions as the form 'symbol' = 'replacing expression'
    :return: ('symbol', 'replacing expression', 'result')
    """
    steps = [] # initialize getSteps
    issubs = True # still substituting
    strkeys,symkeys = [],[]
    for key in subs.keys():
        if isinstance(key,str):
            strkeys.append(key)
        else:
            symkeys.append(key)
    while issubs:
        lookfor = expr.atoms(Symbol)
        issubs = False # after process we are not substituting
        for sym in lookfor:
            name = None
            if sym in symkeys:
                name = sym
            elif str(sym) in strkeys:
                name = str(sym)

            if name:
                newexp = expr.subs(name,subs[name])
                if expr != newexp:
                    steps.append((sym,subs[name],newexp))
                    expr = newexp
                    issubs = True # need to continue substituting
                else:
                    break
    return steps

def print_and_sub(expr, printer=print_latex, **subs):
    """

    :param expr:
    :param printer:
    :param subs:
    :return:
    """
    for key in subs:
        printer(Eq(symbols(key),subs[key]))
    return expr.subs(subs)

def shroud(string,s="$"):
    return "{s}{string}{s}".format(string=string,s=s)

def printableSteps(steps, asEqs = True, converter=latex, shroudwith="$", explanations = True):
    """
    Convert steps obtained from getSteps into printable lines for human readability.

    :param steps: list from getSteps with items of the form ('symbol', 'replacing expression', 'result')
    :param asEqs: convert as equations
    :param converter: converter engine. default is latex
    :return: list of lines
    """
    def Eq(expr1,expr2):
        return converter(expr1)+" = "+ converter(expr2)

    firstexpr,sub,res = steps[0]

    if explanations:
        if len(firstexpr.atoms(Symbol)) == 1:
            lines = ["Procedures to obtain {}:".format(converter(firstexpr)), ""]
        else:
            lines = ["Procedures to obtain the Equation:",""]
    else:
        lines = []

    if sub==res:
        if explanations:
            lines.append("{} can be expressed as:".format(converter(firstexpr)))
        lines.append(shroud(Eq(firstexpr, sub),s=shroudwith))
    else:
        if explanations:
            lines.append("{} can be expressed as:".format(converter(firstexpr)))
        lines.append(shroud(converter(sub),s=shroudwith))
        if explanations:
            lines.append("so we get")
        lines.append(shroud(converter(res),s=shroudwith))

    if asEqs:
        myfunc = lambda x: Eq(firstexpr,x)
    else:
        myfunc = lambda x: converter(x)

    for expr,sub,res in steps[1:]:
        if sub is None:
            if explanations:
                lines.append("rearranging the equation we get:")
            lines.append(shroud(myfunc(res),s=shroudwith))
        else:
            if explanations:
                lines.append("but if we replace:")
            lines.append(shroud(Eq(expr, sub),s=shroudwith))
            if explanations:
                lines.append("we get:")
            lines.append(shroud(myfunc(res),s=shroudwith))

    return lines


def printEquations(eqs, expr = None, inverted= False, printer=print_latex):
    """
    Print equations obtained from getEquations.

    :param expr:
    :param eqs:
    :param inverted:
    :param printer:
    :return:
    """
    if inverted:
        for i in eqs[::-1]:
            printer(i)
        if expr: printer(expr)
    else:
        if expr: printer(expr)
        for i in eqs:
            printer(i)

def isMatrix(sym):
    return isinstance(sym,MatrixBase)

def isMatrixOp(sym):
    """ test if is one of these MatrixSymbol,MatMul,MatAdd,MatPow,Transpose,Inverse
    :param sym:
    :return:
    """
    return isinstance(sym,MatrixExpr)

def isMatrixElement(sym):
    return isinstance(sym,MatrixElement)

def isMatrixSymbol(sym):
    return isinstance(sym,MatrixSymbol)

def isSymbol(sym):
    return isinstance(sym,MatrixExpr)

def getMatrixSymbol(sym, process = latex):
    return process(Matrix(sym))

def show(sym, process = latex, train = [], lavel=0):
    train.append(process(sym))
    if hasattr(sym,"args"):
        for element in sym.args:
            show(element,process, train)
    return train

def placeInMatrix(map, defaul = 0):
    def order(i,j):
        return map.get((i,j),defaul)
    return order

def walkMatrix(matrix,func = ""):
    for element in matrix:
        print element

def printTraversal(expr,traversal = preorder_traversal):
    for i in traversal(expr):
        print i

if __name__ == "__main__":

    A,B = symbols("A,B")
    sx,sy,ox,oy = symbols("sx,sy,ox,oy")
    M1 = Matrix([[Mul(sx,1/ox,evaluate=False),0,0],[0,sy/ox,0],[0,0,1]])
    M1_ = Matrix([[A/B,0,0],[0,A,0],[0,0,1]])
    Ms = MatrixSymbol("M",3,3)
    expr = Matrix(Ms)*M1*M1_
    strlatex=  latex(expr,mul_symbol="dot")
    #printTraversal(expr)

    x = symbols('x', real=True)
    f = symbols('f', real=True)(x)
    v = Matrix([f * sin(x), f * cos(x)])
    v1 = v.diff(x)

    print v1.subs(f, x)
    # prints: Matrix([[x*cos(x) + sin(x)*Derivative(x, x)], [-x*sin(x) + cos(x)*Derivative(x, x)]])
    print v1.subs(f, x).doit()
    # prints: Matrix([[x*cos(x) + sin(x)*Derivative(x, x)], [-x*sin(x) + cos(x)*Derivative(x, x)]])
    print Matrix([e.doit() for e in v1.subs(f, x)])
    # prints: Matrix([[x*cos(x) + sin(x)], [-x*sin(x) + cos(x)]])