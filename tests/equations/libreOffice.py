"""
 There is a simple rule in libreOffice:
    - everything can be shrouded in {} to prevent problems in identifying where an expression begins and ends
    - the brackets are shrouded like so: {left({ expression }right)}
    - operators must have spaces e.g ' + ', ' dot ', ' / ' (some of them does not need but this make things clearer
        and is compatible with the %keywords operators like 'dot')
"""

from sympy.printing.printer import Printer
from sympy.printing.conventions import split_super_sub
from sympy.printing.precedence import precedence
from sympy.core.alphabets import greeks
from sympy.core.function import _coeff_isneg
from sympy.printing.latex import LatexPrinter
import sympy.mpmath.libmp as mlib
from sympy.mpmath.libmp import prec_to_dps
from sympy import Integral, Piecewise, Product, Sum, Add, source
from sympy.core import S, C, Add, Symbol
import re

accepted_latex_functions = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
                    'sinh', 'cosh', 'tanh', 'sqrt', 'ln', 'log', 'sec', 'csc',
                    'cot', 'coth', 're', 'im', 'frac', 'root', 'arg',
                    ]

# Variable name modifiers
modifier_dict = {
    # Accents
    'mathring': lambda s: r'circle '+s,
    'dddot': lambda s: r'dddot '+s,
    'ddot': lambda s: r'ddot '+s,
    'dot': lambda s: r'dot '+s,
    'checkLoaded': lambda s: r'checkLoaded '+s,
    'breve': lambda s: r'breve '+s,
    'acute': lambda s: r'acute '+s,
    'grave': lambda s: r'grave '+s,
    'tilde': lambda s: r'tilde '+s,
    'hat': lambda s: r'hat '+s,
    'bar': lambda s: r'bar '+s,
    'vec': lambda s: r'vec '+s,
    'prime': lambda s: "{"+s+"}'",
    'prm': lambda s: "{"+s+"}'",
    # Faces
    'bold': lambda s: r'bold '+s,
    'bm': lambda s: r'bold '+s,
    'cal': lambda s: r'font <?> {'+s+r'}',
    'scr': lambda s: r'font <?> {'+s+r'}',
    'frak': lambda s: r'font <?> {'+s+r'}',
    # Brackets
    'norm': lambda s: r'ldbracket '+s+' rdbracket',
    'avg': lambda s: r'langle '+s+' rangle',
    'abs': lambda s: r'abs{'+s+r'}',
    'mag': lambda s: r'abs{'+s+r'}',
}

tex_dictionary = {}
greek_letters_set = [i.upper() for i in greeks]+list(greeks)
convert_symbols = frozenset(["i" + i for i in greek_letters_set] + greek_letters_set) # this should work [ "%"+i for i in convert_symbols]
other_symbols = {'aleph', 'hbar'}

def translate(s):
    r'''
    Check for a modifier ending the string.  If present, convert the
    modifier to libreOffice and translate the rest recursively.

    Given a description of a Greek letter or other special character,
    return the appropriate libreOffice symbol.

    Let everything else pass as given.

    >>> translate('alphahatdotprime')
    "{dot hat %alpha}'"
    '''
    # Process the rest
    tex = tex_dictionary.get(s)
    if tex: # if mapped case
        return tex
    elif s in convert_symbols:
        return "%" + s
    else:
        # Process modifiers, if any, and recurse
        for key in sorted(modifier_dict.keys(), key=lambda k:len(k), reverse=True):
            if s.lower().endswith(key) and len(s)>len(key):
                return modifier_dict[key](translate(s[:-len(key)]))
        return s

class libreOfficePrinter(Printer):

    printmethod = "_libreOffice"

    _default_settings = {
        "order": None,
        "mode": "plain",
        "itex": False,
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "long_frac_ratio": 2,
        "mul_symbol": None,
        "inv_trig_style": "abbreviated",
        "mat_str": None,
        "mat_delim": "(",
        "symbol_names": {},
    }

    def __init__(self, settings=None):
        super(libreOfficePrinter,self).__init__(settings)

        if 'mode' in self._settings:
            valid_modes = ['inline', 'plain', 'equation','equation*']
            if self._settings['mode'] not in valid_modes:
                raise ValueError("'mode' must be one of 'inline', 'plain', ""'equation' or 'equation*'")

        if self._settings['fold_short_frac'] is None and self._settings['mode'] == 'inline':
            self._settings['fold_short_frac'] = True

        mul_symbol_table = {None: r" ","ldot": r" cdot ","dot": r" cdot ","times": r" times "}
        self._settings['mul_symbol_latex'] = mul_symbol_table[self._settings['mul_symbol']]
        self._settings['mul_symbol_latex_numbers'] = mul_symbol_table[self._settings['mul_symbol'] or 'dot']
        self._delim_dict = {"(":("{left({","}right)}"),
                            "[":("{left[{","}right]}"),
                            "ldbracket":("{left ldbracket{","}right rdbracket}"),
                            "lbrace":("{left lbrace{","}right rbrace}"),
                            "langle":("{left langle{","}right rangle}"),
                            "lceil":("{left lceil{","}right rceil}"),
                            "lfloor":("{left lfloor{","}right rfloor}"),
                            "lline":("{left lline{","}right rline}"),
                            "ldline":("{left ldline{","}right rdline}")}

    def _do_delim(self, out_str, left_delim = None):
        if self._settings['mat_delim']:
            if left_delim is None:
                left_delim = self._settings['mat_delim']
            left_delim,right_delim = self._delim_dict[left_delim]
            out_str = left_delim + out_str +right_delim
        return out_str

    def _operators(self, op, i="", ifrom=None, ito=None):
        if i: i = "%s=" % i
        if ifrom: op += " from{%s%s}" % (i,ifrom)
        if ito: op += " to{%s}" % ito
        return op

    def parenthesize(self, item, level):
        if precedence(item) <= level:
            return r"{left({%s}right)}" % self._print(item)
        else:
            return self._print(item)

    def _needs_brackets(self, expr):
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed, False otherwise. For example: a + b => True; a => False;
        10 => False; -10 => True.
        """
        return not ((expr.is_Integer and expr.is_nonnegative)
                or (expr.is_Atom and expr is not S.NegativeOne) or expr.is_Matrix)

    def _needs_function_brackets(self, expr):
        """
        Returns True if the expression needs to be wrapped in brackets when
        passed as an argument to a function, False otherwise. This is a more
        liberal version of _needs_brackets, in that many expressions which need
        to be wrapped in brackets when added/subtracted/raised to a power do
        not need them when passed to a function. Such an example is a*b.
        """
        if not self._needs_brackets(expr):
            return False
        else:
            # Muls of the form a*b*c... can be folded
            if expr.is_Mul and not self._mul_is_clean(expr):
                return True
            # Pows which don't need brackets can be folded
            elif expr.is_Pow and not self._pow_is_clean(expr):
                return True
            # Add and Function always need brackets
            elif expr.is_Add or expr.is_Function:
                return True
            else:
                return False

    def _needs_mul_brackets(self, expr, last=False):
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of a Mul, False otherwise. This is True for Add,
        but also for some container objects that would not need brackets
        when appearing last in a Mul, e.g. an Integral. ``last=True``
        specifies that this expr is the last to appear in a Mul.
        """
        return expr.is_Add or (not last and
            any([expr.has(x) for x in (Integral, Piecewise, Product, Sum)]))

    def _mul_is_clean(self, expr):
        for arg in expr.args:
            if arg.is_Function:
                return False
        return True

    def _pow_is_clean(self, expr):
        return not self._needs_brackets(expr.base)

    def _do_exponent(self, expr, exp):
        if exp is not None:
            return r"%s^{%s}" % (self._do_delim(expr), exp)
        else:
            return expr

    def _print_bool(self, e):
        return r"{%s}" % e

    _print_BooleanTrue = _print_bool
    _print_BooleanFalse = _print_bool

    def _print_NoneType(self, e):
        return r"{%s}" % e

    def _print_Add(self, expr, order=None):
        if self.order == 'none':
            terms = list(expr.args)
        else:
            terms = self._as_ordered_terms(expr, order=order)
        tex = self._print(terms[0])

        for term in terms[1:]:
            if not _coeff_isneg(term):
                tex += " + " + self._print(term)
            else:
                tex += " - " + self._print(-term)

        return tex

    def _print_Float(self, expr):
        # Based off of that in StrPrinter
        dps = prec_to_dps(expr._prec)
        str_real = mlib.to_str(expr._mpf_, dps, strip_zeros=True)

        # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
        # thus we use the number separator
        separator = self._settings['mul_symbol_latex_numbers']

        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]

            return r"%s%s10^{%s}" % (mant, separator, exp)
        elif str_real == "+inf":
            return r"{infinity}"
        elif str_real == "-inf":
            return r"{-infinity}"
        else:
            return str_real

    def _print_Mul(self, expr):
        coeff, _ = expr.as_coeff_Mul()

        if not coeff.is_negative:
            tex = ""
        else:
            expr = -expr
            tex = "- "

        from sympy.simplify import fraction
        numer, denom = fraction(expr, exact=True)
        separator = self._settings['mul_symbol_latex']
        numbersep = self._settings['mul_symbol_latex_numbers']

        def convert(expr):
            if not expr.is_Mul:
                return str(self._print(expr))
            else:
                _tex = last_term_tex = ""

                if self.order not in ('old', 'none'):
                    args = expr.as_ordered_factors()
                else:
                    args = expr.args

                for i, term in enumerate(args):
                    term_tex = self._print(term)

                    if self._needs_mul_brackets(term, last=(i == len(args) - 1)):
                        term_tex = self._do_delim(term_tex)

                    if re.search("[0-9][} ]*$", last_term_tex) and \
                            re.match("[{ ]*[-+0-9]", term_tex):
                        # between two numbers
                        _tex += numbersep
                    elif _tex:
                        _tex += separator

                    _tex += term_tex
                    last_term_tex = term_tex
                return _tex

        if denom is S.One:
            # use the original expression here, since fraction() may have
            # altered it when producing numer and denom
            tex += convert(expr)
        else:
            snumer = convert(numer)
            sdenom = convert(denom)
            ldenom = len(sdenom.split()) # pow
            ratio = self._settings['long_frac_ratio']
            if self._settings['fold_short_frac'] and ldenom <= 2 and not "^" in sdenom:
                # handle short fractions
                if self._needs_mul_brackets(numer, last=False):
                    tex += self._do_frac(self._do_delim(snumer), sdenom)
                else:
                    tex += self._do_frac(snumer, sdenom)
            elif len(snumer.split()) > ratio*ldenom:
                # handle long fractions
                if self._needs_mul_brackets(numer, last=True):
                    tex += r"%s%s%s" % (self._do_frac(1, sdenom), separator, self._do_delim(snumer))
                elif numer.is_Mul:
                    # split a long numerator
                    a = S.One
                    b = S.One
                    for x in numer.args:
                        if self._needs_mul_brackets(x, last=False) or len(convert(a*x).split()) > ratio*ldenom or (b.is_commutative is x.is_commutative is False):
                            b *= x
                        else:
                            a *= x
                    if self._needs_mul_brackets(b, last=True):
                        tex += r"%s%s%s" % (self._do_frac(convert(a), sdenom), separator, self._do_delim(convert(b)))
                    else:
                        tex += r"%s%s%s" % (self._do_frac(convert(a), sdenom), separator, convert(b))
                else:
                    tex += r"%s%s%s" % (self._do_frac(1, sdenom), separator, snumer)
            else:
                tex += self._do_frac(snumer, sdenom)
        return tex

    def _do_frac(self,numer,denom):
        # TODO: support: over,wideslash,/,div,wideslash e.g. {<?>} wideslash {<?>} a / b a div b {<?>} wideslash {<?>}
        if self._settings['fold_short_frac']:
            return r"%s / %s" % (numer,denom)
        else:
            return r"{%s} over {%s}" % (numer,denom)

    def _print_Pow(self, expr):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational and abs(expr.exp.p) == 1 and expr.exp.q != 1:
            base = self._print(expr.base)
            expq = expr.exp.q

            if expq == 2:
                tex = r"sqrt{%s}" % base
            else:
                tex = r"nroot{%d}{%s}" % (expq, base)
            if expr.exp.is_negative:
                return r"{1} over {%s}" % tex
            else:
                return tex
        elif self._settings['fold_frac_powers'] and expr.exp.is_Rational and expr.exp.q != 1:
            base, p, q = self._print(expr.base), expr.exp.p, expr.exp.q
            if expr.base.is_Function:
                return self._print(expr.base, self._do_frac(p, q))
            if self._needs_brackets(expr.base):
                return self._do_exponent(base,self._do_frac(p, q)) # r" left(%s right)^{%s/%s}" % (base, p, q)
            return r"%s^{%s}" % (base, self._do_frac(p, q))
        elif expr.exp.is_Rational and expr.exp.is_negative and expr.base.is_commutative:
            # Things like 1/x
            return self._print_Mul(expr)
        else:
            if expr.base.is_Function:
                return self._print(expr.base, self._print(expr.exp))
            else:
                if expr.is_commutative and expr.exp == -1:
                    #solves issue 4129
                    #As Mul always simplify 1/x to x**-1
                    #The objective is achieved with this hack
                    #first we get the latex for -1 * expr,
                    #which is a Mul expression
                    tex = self._print(S.NegativeOne * expr).strip()
                    #the result comes with a minus and a space, so we remove
                    if tex[:1] == "-":
                        return tex[1:].strip()
                if self._needs_brackets(expr.base):
                    tex = self._do_exponent("%s","%s")# r" left({%s} right)^{%s}"
                    return tex % (self._print(expr.base),self._print(expr.exp))
                else:
                    tex = r"%s^{%s}"
                return tex % (self._print(expr.base),self._print(expr.exp))

    def _print_Sum(self, expr): # TODO
        tex = ""

        for lim in reversed(expr.limits): # get each integral limits in
            tex += self._operators(r"sum", *[self._print(l) for l in lim])

        if isinstance(expr.function, Add):
            tex += self._do_delim(self._print(expr.function))
        else:
            tex += self._print(expr.function)

        return tex

    def _print_Product(self, expr): # TODO
        if len(expr.limits) == 1:
            tex = r"\prod_{%s=%s}^{%s} " % \
                tuple([ self._print(i) for i in expr.limits[0] ])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\prod_{\substack{%s}} " % \
                str.join('\\\\', [ _format_ineq(l) for l in expr.limits ])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex

    def _print_BasisDependent(self, expr): # TODO
        from sympy.vector import Vector

        o1 = []
        if expr == expr.zero:
            return expr.zero._latex_form
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]

        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key = lambda x:x[0].__str__())
            for k, v in inneritems:
                if v == 1:
                    o1.append(' + ' + k._latex_form)
                elif v == -1:
                    o1.append(' - ' + k._latex_form)
                else:
                    arg_str = '(' + libreOfficePrinter().doprint(v) + ')'
                    o1.append(' + ' + arg_str + k._latex_form)

        outstr = (''.join(o1))
        if outstr[1] != '-':
            outstr = outstr[3:]
        else:
            outstr = outstr[1:]
        return outstr

    def _print_Indexed(self, expr): # TODO
        tex = self._print(expr.base)+'_{%s}' % ','.join(map(self._print, expr.indices))
        return tex

    def _print_IndexedBase(self, expr): # TODO
        return self._print(expr.label)

    def _print_Derivative(self, expr): # TODO
        dim = len(expr.variables)
        if requires_partial(expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = r'd'


        if dim == 1:
            tex = r"\frac{%s}{%s %s}" % (diff_symbol, diff_symbol,
                self._print(expr.variables[0]))
        else:
            multiplicity, i, tex = [], 1, ""
            current = expr.variables[0]

            for symbol in expr.variables[1:]:
                if symbol == current:
                    i = i + 1
                else:
                    multiplicity.append((current, i))
                    current, i = symbol, 1
            else:
                multiplicity.append((current, i))

            for x, i in multiplicity:
                if i == 1:
                    tex += r"%s %s" % (diff_symbol, self._print(x))
                else:
                    tex += r"%s %s^{%s}" % (diff_symbol, self._print(x), i)

            tex = r"\frac{%s^{%s}}{%s} " % (diff_symbol, dim, tex)

        if isinstance(expr.expr, C.AssocOp):
            return r"%s\left(%s\right)" % (tex, self._print(expr.expr))
        else:
            return r"%s %s" % (tex, self._print(expr.expr))

    def _print_Subs(self, subs): # TODO
        expr, old, new = subs.args
        latex_expr = self._print(expr)
        latex_old = (self._print(e) for e in old)
        latex_new = (self._print(e) for e in new)
        latex_subs = r'\\ '.join(
            e[0] + '=' + e[1] for e in zip(latex_old, latex_new))
        return r'\left. %s \right|_{\substack{ %s }}' % (latex_expr, latex_subs)

    def _print_Integral(self, expr):
        tex, symbols = "", []

        # Only up to \iiint exists
        if len(expr.limits) <= 3 and all(len(lim) == 1 for lim in expr.limits):
            # Use len(expr.limits)-1 so that syntax highlighters don't think
            # \" is an escaped quote
            tex = r"i" + "i"*(len(expr.limits) - 1) + "nt" # get int until iiint
            symbols = [r"d%s" % self._print(symbol[0]) for symbol in expr.limits] # get symbol domains
        else:
            for lim in reversed(expr.limits): # get each integral limits in reversed order
                symbol = lim[0] # get symbol domains
                # domain = lim[0],lfrom = lim[1],lto = lim[2]
                if len(lim) > 1:
                    if len(lim) == 3:
                        tex += self._operators(r"int", None, self._print(lim[1]), self._print(lim[2]))
                    if len(lim) == 2:
                        tex += self._operators(r"int", None, self._print(lim[1]))
                else:
                    tex += self._operators(r"int")

                symbols.insert(0, r"d%s" % self._print(symbol))

        return r"%s %s %s" % (tex,str(self._print(expr.function)), "".join(symbols))

    def _print_Limit(self, expr): # TODO
        e, z, z0, dir = expr.args

        tex = r"\lim_{%s \to " % self._print(z)
        if z0 in (S.Infinity, S.NegativeInfinity):
            tex += r"%s}" % self._print(z0)
        else:
            tex += r"%s^%s}" % (self._print(z0), self._print(dir))

        if isinstance(e, C.AssocOp):
            return r"%s\left(%s\right)" % (tex, self._print(e))
        else:
            return r"%s %s" % (tex, self._print(e))

    def _print_Function(self, expr, exp=None):
        '''
        Render functions to LaTeX, handling functions that LaTeX knows about
        e.g., sin, cos, ... by using the proper LaTeX command (\sin, \cos, ...).
        For single-letter function names, render them as regular LaTeX math
        symbols. For multi-letter function names that LaTeX does not know
        about, (e.g., Li, sech) use \operatorname{} so that the function name
        is rendered in Roman font and LaTeX handles spacing properly.

        expr is the expression involving the function
        exp is an exponent
        '''
        func = expr.func.__name__

        if hasattr(self, '_print_' + func):
            return getattr(self, '_print_' + func)(expr, exp)
        else:
            args = [ str(self._print(arg)) for arg in expr.args ]
            # How inverse trig functions should be displayed, formats are:
            # abbreviated: asin, full: arcsin, power: sin^-1
            inv_trig_style = self._settings['inv_trig_style']
            # If we are dealing with a power-style inverse trig function
            inv_trig_power_case = False
            # If it is applicable to fold the argument brackets
            do_not_group = self._settings['fold_func_brackets'] and \
                len(args) == 1 and not self._needs_function_brackets(expr.args[0])

            inv_trig_table = ["asin", "acos", "atan", "acot"]

            # If the function is an inverse trig function, handle the style
            if func in inv_trig_table:
                if inv_trig_style == "abbreviated":
                    func = func
                elif inv_trig_style == "full":
                    func = "arc" + func[1:]
                elif inv_trig_style == "power":
                    func = func[1:]
                    inv_trig_power_case = True

                    # Can never fold brackets if we're raised to a power
                    if exp is not None:
                        do_not_group = False

            if inv_trig_power_case:
                name = r"{%s}^{-1}" % func
            elif exp is not None:
                name = r'{%s}^{%s}' % (func, exp)
            else:
                name = func

            if not do_not_group:
                name += self._do_delim("%s" % ",".join(args), "(")
            else:
                name += "%s" % ",".join(args)

            if inv_trig_power_case and exp is not None:
                name += r"^{%s}" % exp

            return name

    # _print_UndefinedFunction(self, expr):  # TODO

    #_print_FunctionClass(self, expr) # TODO

    # _print_Lambda(self, expr): # TODO

    # _print_Min(self, expr, exp=None): # TODO

    # _print_Max(self, expr, exp=None):  # TODO

    # _print_floor(self, expr, exp=None):  # TODO

    # _print_ceiling(self, expr, exp=None):  # TODO

    def _print_Abs(self, expr, exp=None):
        tex = self._do_delim(self._print(expr.args[0]),"lline")
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex
    _print_Determinant = _print_Abs

    ################################################# UNORDERED ############################

    def _print_Relational(self, expr):
        if self._settings['itex']:
            mo = " >> "
            le = " << "
            gt = " geslant "
            lt = " leslant "
        else:
            mo = " > "
            le = " <"
            gt = " >= "
            lt = " <= "

        charmap = {"==": " = ",">": mo,"<": le,">=": gt,"<=": lt,"!=": r" <> "}

        return "%s%s%s" % (self._print(expr.lhs),charmap[expr.rel_op], self._print(expr.rhs))

    def _print_Symbol(self, expr):
        if expr in self._settings['symbol_names']:
            return self._settings['symbol_names'][expr]

        return self._deal_with_super_sub(expr.name)

    _print_RandomSymbol = _print_Symbol
    _print_MatrixSymbol = _print_Symbol

    def _deal_with_super_sub(self, string):

        name, supers, subs = split_super_sub(string)

        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]

        # glue all items together:
        if len(supers) > 0:
            name += "^{%s}" % " ".join(supers)
        if len(subs) > 0:
            name += "_{%s}" % " ".join(subs)

        return name

    def _print_Rational(self, expr):
        if expr.q != 1:
            sign = ""
            p = expr.p
            if expr.p < 0:
                sign = "- "
                p = -p
            return r"%s%s" % (sign, self._do_frac(p, expr.q))
        else:
            return self._print(expr.p)

    #################################### MATRICES ############################################

    def _print_MatrixBase(self, expr):
        lines = []
        for line in range(expr.rows):  # horrible, should be 'rows'
            lines.append(" # ".join([ self._print(i) for i in expr[line, :] ]))
        return self._do_delim(r'{matrix{ %s }}' % r" ## ".join(lines))

    _print_ImmutableMatrix = _print_MatrixBase
    _print_Matrix = _print_MatrixBase

    def _print_MatrixElement(self, expr):
        return self._print(expr.parent) + '_{%s,%s}'%(expr.i, expr.j)

    def _print_MatrixSlice(self, expr):
        def latexslice(x):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[1] == x[0] + 1:
                del x[1]
            if x[0] == 0:
                x[0] = ''
            return ':'.join(map(self._print, x))
        return (self._print(expr.parent) + self._do_delim(latexslice(expr.rowslice) + ', ' + latexslice(expr.colslice)))

    def _print_BlockMatrix(self, expr):
        return self._print(expr.blocks)

    def _print_Transpose(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"%s^{T}" % self._do_delim(self._print(mat))
        else:
            return "%s^{T}" % self._print(mat)

    def _print_Adjoint(self, expr): # see Hermitian adjoint # see conjugate transpose
        mat = expr.arg
        from sympy.matrices import MatrixSymbol
        if not isinstance(mat, MatrixSymbol):
            return r"%s" % self._do_delim(self._print(mat)) + "^{%Ux2020}"
        else:
            return "%s" % self._print(mat) + "^{%Ux2020}"

    def _print_MatAdd(self, expr):
        terms = list(expr.args)
        tex = " + ".join(map(self._print, terms))
        return tex

    def _print_MatMul(self, expr):
        from sympy import Add, MatAdd, HadamardProduct

        def parens(x):
            if isinstance(x, (Add, MatAdd, HadamardProduct)):
                return r"{%s}" % self._do_delim(self._print(x))
            return self._print(x)
        return ' '.join(map(parens, expr.args))

    def _print_MatPow(self, expr):
        base, exp = expr.base, expr.exp
        from sympy.matrices import MatrixSymbol
        if not isinstance(base, MatrixSymbol):
            return r"%s^{%s}" % (self._do_delim(self._print(base)), self._print(exp))
        else:
            return "%s^{%s}" % (self._print(base), self._print(exp))

    def _print_ZeroMatrix(self, Z):
        return r"{bold 0}"

    def _print_Identity(self, I):
        return r"{bold I}"

def libreOffice(expr, **settings):
    return libreOfficePrinter(settings).doprint(expr)

def print_libreOffice(expr, **settings):
    """Prints libreOffice representation of the given expression."""
    print(libreOffice(expr, **settings))

if __name__ == "__main__":


    from sympy import latex, symbols, Matrix, MatrixSymbol, Eq, S, collect, Pow
    from sympy.functions import sin,cos, asinh, asin

    Aow,Aoh = symbols("W_oA,H_oA")
    Asw,Ash = symbols("W_sA,H_sA")
    Bow,Boh = symbols("W_oB,H_oB")
    Bsw,Bsh = symbols("W_sB,H_sB")
    oAToB,oATsA,sBToB = symbols("oAToB,oATsA,sBToB")

    _oATsA = Matrix([[Asw/Aow,0,0],[0,Ash/Aoh,0],[0,0,1]])
    sATsB = MatrixSymbol("M",3,3)
    _sATsB = Matrix(sATsB)
    _sBToB = Matrix([[Bow/Bsw,0,0],[0,Boh/Bsh,0],[0,0,1]])
    #print libreOffice(Eq(oAToB, sBToB*sTM*oATsA))
    #print libreOffice(Eq(oAToB, sBToB_mat*sTM*oATsA_mat))
    print libreOffice(_sBToB*_sATsB*_oATsA,inv_trig_style="power")
    #print libreOffice(Pow(Aow,3),inv_trig_style="power")
    #print libreOffice(Pow(asin(sBToB_mat*sTM_mat*oATsA_mat, evaluate = False),3,evaluate=False),inv_trig_style="power")
    #print
    #print libreOffice(asin(Pow(sBToB_mat*sTM_mat*oATsA_mat,2,evaluate=False), evaluate = False),inv_trig_style="power")

    """
    from sympy.functions import adjoint
    A = MatrixSymbol('A', 3, 5)
    B = MatrixSymbol('B', 5, 3)
    print libreOffice(adjoint(A*B))"""