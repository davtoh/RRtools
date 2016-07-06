from sympy.utilities.lambdify import lambdify #, implemented_function
from sympy.tensor import IndexedBase, Indexed
from sympy import Sum

def test_vars_in_expr(expr, vars, once = False):
    """
    Test variables if exist in expression.

    :param expr: expression (with variables).
    :param vars: list of symbolic variables.
    :param once: True to return first variable not in expression,
                False to return a list of all variables that failed the test.
    :return: first variable or list of variables.
    """
    vars1 = [str(x) for x in vars]
    vars2 = frozenset([str(x) for x in expr.atoms()])
    vars = []
    for v in vars1:
        if v not in vars2:
            if once: return v # just return variable
            vars.append(v) # else append to list
    return vars

def evalSum(vars, expr, **kwargs):
    """
    Evaluate and get function of expression with summation (one of the ways to evaluate summation) support.
    :param vars: symbolic variables in expression as well as the order of arguments in output function.
    :param expr: expression to get function.
    :param kwargs: options (Not implemented) or substitutions in expression.
    :return: function (anonymous or defined).
    """
    if kwargs:
        myeval = expr.subs(kwargs).doit()
    else:
        myeval = expr.doit()

    dummies = test_vars_in_expr(expr,vars)
    if dummies:
        Warning("{} is/are not in the expression".format(dummies))

    ivars = list(expr.atoms(IndexedBase))
    indexes,vectors = [],[]
    for i in ivars:
        try:
            indexes.append(vars.index(i))
            vectors.append(i)
        except ValueError:
            pass
    #pos = [isinstance(var,(IndexedBase)) for var in vars] # get positions
    #indexes = [ind.args[1] for ind in list(myeval.atoms(Indexed))] # get indexes
    #expr.limits[0].args.index(vars) # index,l1,l2 = expr.limits[0]
    def getVariables(newexpr,args):
        maps2 = {}
        vectors2 = [str(i) for i in vectors]
        for query in newexpr.atoms(Indexed):
            try:
                vector,index = query.args
                maps2[query] = args[indexes[vectors2.index(str(vector))]][index]
            except Exception as e:
                pass
        return maps2

    def getVariables_(newexpr,args):
        # TODO: this is intended to replace getVariables for better performance
        # FIXME not working
        maps2 = {}
        mymap = dict(zip(indexes,vectors))
        for index, var in mymap.iteritems():
            vec = args[index]
            for i,val in enumerate(vec):
                maps2[var[i]]=val
        return maps2

    if myeval.atoms(Sum): # Not ready
        def myfunc(*args): # if other variables
            maps = {vars[i]:val for i,val in enumerate(args) if i not in indexes}
            newexpr = expr.subs(maps).doit()
            maps2 = getVariables(newexpr,args)
            return newexpr.subs(maps2)

    elif indexes: # is ready Indexed
        if len(indexes) == len(vars): # just replace vectors
            def myfunc(*args):
                return expr.doit().subs(getVariables(expr,args))
        else: # replace vetors and other variables
            def myfunc(*args):
                maps = getVariables(expr,args)
                maps.update({vars[i]:val for i,val in enumerate(args) if i not in indexes}) # user variables are priority
                return expr.doit().subs(maps)

    else: # nothing especial, just lambdify it
        myfunc = lambdify(vars,expr,**kwargs)
    return myfunc