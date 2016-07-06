"""
developing equations... not yet realized
"""
from sympy import symbols, Function, S, Eq, IndexedBase, Indexed, Idx
from sympy.printing import latex, print_latex
from sympy.utilities.lambdify import lambdify, implemented_function
from printer import print_libreOffice, getSteps, getEqs, printableSteps
from sympy.functions import exp
from sympy.abc import _clash
#from sympy.utilities.source import get_class,get_mod_func,source as print_source
from RRtoolbox.lib.inspector import funcData, inspect, load, reloadFunc

x,y,z,c,u,v,w = symbols("x,y,z,c,u,v,w")
n, m, i, j = symbols('n m i j', integer=True) # indexes and dimensions
n1, m1, n2, m2 = symbols('n_1 m_1 n_2 m_2', integer=True)
img1 = IndexedBase('img_1', shape=(n1,m1))
img2 = IndexedBase('img_2', shape=(n2,m2))

def test():
    """
    test to see how lamdify and Function work together
    :return:
    """
    def scale(x,y): # test function
        return x+y
    """
    some commend here
    """
    f = Function("scale")(x,y)
    f2 = lambdify((x,y),f,{"scale":scale})
    print f2(1,2)

### FUNCTIONS ####
from RRtoolbox.lib.arrayops.convert import getSOpointRelation
from RRtoolbox.lib.arrayops.filters import sigmoid,normsigmoid

def getSOpointRelation_LINES(img1=img1,img2=img2):
    """
    Return parameters to change scaled point to original point.

    destine_domain = relation*source_domain.

    :param source_shape: image shape for source domain
    :param destine_shape: image shape for destine domain
    :return: x, y coordinate relations

    .. note:: Used to get relations to convert scaled points to original points of an Image.
    """
    lines = []
    lines.append("given to images {} and {}".format(img1,img2))
    relation = getSOpointRelation(img1.shape,img2.shape) # funcData(getSOpointRelation)
    print img1.shape,img2.shape, relation
    return lines

def sigmoid_LINES(converter=latex):
    x, alpha, beta, max, min, sigmoid = symbols("x alpha beta x_max x_min sigmoid")
    result = Eq(sigmoid, (max-min) * (1 / (exp((beta - x) / alpha) + 1)) + min)
    print latex(result)
    print latex(S("(x_max-x_min) * (1 / (exp((beta - x) / alpha) + 1)) + x_min",locals = _clash))

if __name__ == "__main__":
    lines = getSOpointRelation_LINES()
    print "\n".join(lines)

