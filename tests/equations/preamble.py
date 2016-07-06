import numpy as np
from sympy import Matrix, MatrixSymbol, Eq, latex, symbols,eye, IndexedBase, Indexed, Idx
from sympy.functions import sin,cos
from sympy.parsing.sympy_parser import parse_expr

n, m, i, j = symbols('n m i j', integer=True) # indexes and dimensions
x,y,z,c,u,v,w = symbols("x,y,z,c,u,v,w") # axis symbols
alpha_x,alpha_y,alpha_z,alpha_u,alpha_v,alpha_w = symbols("alpha_x,alpha_y,alpha_z,alpha_u,alpha_v,alpha_w") # axis scaling
theta_x, theta_y, theta_z, theta_u, theta_v, theta_w = symbols("theta_x,theta_y,theta_z,theta_u,theta_v,theta_w") # axis rotations
beta,theta,delta,gamma,phi = symbols("beta,theta,delta,gamma,phi") # angles
image = IndexedBase('img', shape=(n,m)).shape # Matrices

def adecuate2for(a):
    if isinstance(a,dict):
        return a.iteritems()
    else:
        return a

def replaceIndexes(expr, variables, usedelim=None, delim ="({})", strfy = str):
    """

    :param expr:
    :param variables:
    :param usedelim:
    :param delim:
    :param strfy:
    :return:
    """
    mystr = strfy(expr)
    for key,val in adecuate2for(variables):
        key = strfy(key)
        try:
            if usedelim is True or usedelim is None and val < 0:
                if callable(delim):
                    val = delim(val)
                else:
                    val = delim.format(val)
        finally:
            val = strfy(val) # ensures delim
        if key in mystr:
            mystr = mystr.replace(key,val)
    return mystr #parse_expr(mystr,evaluate=False)

def indexMap(symbol,arr):
    return [(str(symbol[i]),arr[i]) for i in xrange(len(arr))]

def convert_substitutions(variables, printfunc = latex, joining = "\wedge "):
    a = ["{} = {}".format(printfunc(key),printfunc(val)) for key,val in adecuate2for(variables)]
    return joining.format(joining).join(a)