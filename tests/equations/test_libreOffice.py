# /home/davtoh/.local/lib/python2.7/site-packages/sympy/printing/tests
# http://docs.sympy.org/dev/modules/printing.html
# http://docs.sympy.org/latest/tutorial/printing.html
# http://docs.sympy.org/latest/tutorial/matrices.html
# http://docs.sympy.org/0.7.2/modules/matrices/expressions.html
# http://web.cs.iastate.edu/~cs577/handouts/homogeneous-transform.pdf
# how to convert to numpy func http://stackoverflow.com/a/10683911/5288758
# generate code http://docs.sympy.org/dev/modules/utilities/codegen.html
# some examples https://github.com/sympy/sympy/wiki/Quick-examples

from sympy import (
    Abs, Chi, Ci, CosineTransform, Dict, Ei, Eq, FallingFactorial, FiniteSet,
    Float, FourierTransform, Function, IndexedBase, Integral, Interval,
    InverseCosineTransform, InverseFourierTransform,
    InverseLaplaceTransform, InverseMellinTransform, InverseSineTransform,
    Lambda, LaplaceTransform, Limit, Matrix, Max, MellinTransform, Min, Mul,
    Order, Piecewise, Poly, ring, field, ZZ, Pow, Product, Range, Rational,
    RisingFactorial, RootOf, RootSum, S, Shi, Si, SineTransform, Subs,
    Sum, Symbol, ImageSet, Tuple, Union, Ynm, Znm, arg, asin,
    assoc_laguerre, assoc_legendre, binomial, catalan, ceiling, Complement,
    chebyshevt, chebyshevu, conjugate, cot, coth, diff, dirichlet_eta,
    exp, expint, factorial, factorial2, floor, gamma, gegenbauer, hermite,
    hyper, im, im, jacobi, laguerre, legendre, lerchphi, log, lowergamma,
    meijerg, oo, polar_lift, polylog, re, re, root, sin, sqrt, symbols,
    uppergamma, zeta, subfactorial, totient, elliptic_k, elliptic_f,
    elliptic_e, elliptic_pi, cos, tan, Wild, true, false, Equivalent, Not,
    Contains, divisor_sigma)

from libreOffice import libreOffice, translate
from sympy.printing import latex
from sympy.utilities.pytest import XFAIL, raises
from sympy.functions import DiracDelta, Heaviside, KroneckerDelta, LeviCivita
from sympy.logic import Implies
from sympy.logic.boolalg import And, Or, Xor
from sympy.core.trace import Tr

x, y, z, t, a, b, mu, tau = symbols('x y z t a b mu tau')
k, m, n = symbols('k m n', integer=True)


def test_printmethod():
    class R(Abs):
        def _latex(self, printer):
            return "foo(%s)" % printer._print(self.args[0])
    assert libreOffice(R(x)) == "foo(x)"

    class R(Abs):
        def _latex(self, printer):
            return "foo"
    assert libreOffice(R(x)) == "foo"


def test_latex_basic():
    assert libreOffice(1 + x) == "x + 1"
    assert libreOffice(x ** 2) == "x^{2}"
    assert libreOffice(x ** (1 + x)) == "x^{x + 1}"
    assert libreOffice(x ** 3 + x + 1 + x ** 2) == "x^{3} + x^{2} + x + 1"

    assert libreOffice(2 * x * y) == "2 x y"
    assert libreOffice(2 * x * y, mul_symbol='dot') == r"2 cdot x cdot y"

    assert libreOffice(1 / x) == r"{1} over {x}"
    assert libreOffice(1 / x, fold_short_frac=True) == "1 / x"
    assert libreOffice(1 / x ** 2) == r"{1} over {x^{2}}"
    assert libreOffice(x / 2) == r"{x} over {2}"
    assert libreOffice(x / 2, fold_short_frac=True) == "x / 2"
    assert libreOffice((x + y) / (2 * x)) == r"{x + y} over {2 x}"
    assert libreOffice((x + y) / (2 * x), fold_short_frac=True) == r'{left({x + y}right)} / 2 x'
    assert libreOffice((x + y) / (2 * x), long_frac_ratio=0) == r'{1} over {2 x} {left({x + y}right)}'
    assert libreOffice((x + y) / x) == r'{1} over {x} {left({x + y}right)}'
    assert libreOffice((x + y) / x, long_frac_ratio=3) == r'{x + y} over {x}'

    assert libreOffice(2 * Integral(x, (x, a, b)) / 3) == r'{2} over {3} int from{a} to{b} x dx' # int from 0 to x f(t) dt
    assert libreOffice(2 * Integral(x, (x, a, b)) / 3, fold_short_frac=True) == r'{left({2 int from{a} to{b} x dx}right)} / 3'

    assert libreOffice(2 * Integral(x, x) / 3) == r'{2} over {3} int x dx' # int from 0 to x f(t) dt
    assert libreOffice(2 * Integral(x, x) / 3, fold_short_frac=True) == r'{left({2 int x dx}right)} / 3'
    assert libreOffice(sqrt(x)) == r"sqrt{x}"
    assert libreOffice(x ** Rational(1, 3)) == r'nroot{3}{x}'
    assert libreOffice(sqrt(x) ** 3) == r'x^{{3} over {2}}'
    assert libreOffice(sqrt(x), itex=True) == r"sqrt{x}" # ????
    assert libreOffice(x ** Rational(1, 3), itex=True) == r"nroot{3}{x}"
    assert libreOffice(sqrt(x) ** 3, itex=True) == r'x^{{3} over {2}}'
    assert libreOffice(x ** Rational(3, 4)) == r'x^{{3} over {4}}'
    assert libreOffice(x ** Rational(3, 4), fold_frac_powers=True) == r'x^{{3} over {4}}'
    assert libreOffice((x + 1) ** Rational(3, 4)) == r'{left({x + 1}right)}^{{3} over {4}}'
    assert libreOffice((x + 1) ** Rational(3, 4), fold_frac_powers=True) == r'{left({x + 1}right)}^{{3} over {4}}'

    assert libreOffice(1.5e20 * x) == r"1.5 \cdot 10^{20} x"
    assert libreOffice(1.5e20 * x, mul_symbol='dot') == r"1.5 \cdot 10^{20} \cdot x"
    assert libreOffice(1.5e20 * x, mul_symbol='times') == r"1.5 \times 10^{20} \times x"

    assert libreOffice(1 / sin(x)) == r"\frac{1}{\sin{\left (x \right )}}"
    assert libreOffice(sin(x) ** -1) == r"\frac{1}{\sin{\left (x \right )}}"
    assert libreOffice(sin(x) ** Rational(3, 2)) == r"\sin^{\frac{3}{2}}{\left (x \right )}"
    assert libreOffice(sin(x) ** Rational(3, 2), fold_frac_powers=True) == r"\sin^{3/2}{\left (x \right )}"

    assert libreOffice(~x) == r"\neg x"
    assert libreOffice(x & y) == r"x \wedge y"
    assert libreOffice(x & y & z) == r"x \wedge y \wedge z"
    assert libreOffice(x | y) == r"x \vee y"
    assert libreOffice(x | y | z) == r"x \vee y \vee z"
    assert libreOffice((x & y) | z) == r"z \vee \left(x \wedge y\right)"
    assert libreOffice(Implies(x, y)) == r"x \Rightarrow y"
    assert libreOffice(~(x >> ~y)) == r"x \not\Rightarrow \neg y"
    assert libreOffice(Implies(Or(x, y), z)) == r"\left(x \vee y\right) \Rightarrow z"
    assert libreOffice(Implies(z, Or(x, y))) == r"z \Rightarrow \left(x \vee y\right)"

    assert libreOffice(~x, symbol_names={x: "x_i"}) == r"\neg x_i"
    assert libreOffice(x & y, symbol_names={x: "x_i", y: "y_i"}) == r"x_i \wedge y_i"
    assert libreOffice(x & y & z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == r"x_i \wedge y_i \wedge z_i"
    assert libreOffice(x | y, symbol_names={x: "x_i", y: "y_i"}) == r"x_i \vee y_i"
    assert libreOffice(x | y | z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == r"x_i \vee y_i \vee z_i"
    assert libreOffice((x & y) | z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == r"z_i \vee \left(x_i \wedge y_i\right)"
    assert libreOffice(Implies(x, y), symbol_names={x: "x_i", y: "y_i"}) == r"x_i \Rightarrow y_i"


def test_latex_builtins():
    assert libreOffice(True) == r"\mathrm{True}"
    assert libreOffice(False) == r"\mathrm{False}"
    assert libreOffice(None) == r"\mathrm{None}"
    assert libreOffice(true) == r"\mathrm{True}"
    assert libreOffice(false) == r'\mathrm{False}'

def test_latex_Float():
    assert libreOffice(Float(1.0e100)) == r"1.0 \cdot 10^{100}"
    assert libreOffice(Float(1.0e-100)) == r"1.0 \cdot 10^{-100}"
    assert libreOffice(Float(1.0e-100), mul_symbol="times") == r"1.0 \times 10^{-100}"
    assert libreOffice(1.0 * oo) == r"\infty"
    assert libreOffice(-1.0 * oo) == r"- \infty"


def test_latex_symbols():
    Gamma, lmbda, rho = symbols('Gamma, lambda, rho')
    mass, volume = symbols('mass, volume')
    assert libreOffice(Gamma + lmbda) == r"\Gamma + \lambda"
    assert libreOffice(Gamma * lmbda) == r"\Gamma \lambda"
    assert libreOffice(Symbol('q1')) == r"q_{1}"
    assert libreOffice(Symbol('q21')) == r"q_{21}"
    assert libreOffice(Symbol('epsilon0')) == r"\epsilon_{0}"
    assert libreOffice(Symbol('omega1')) == r"\omega_{1}"
    assert libreOffice(Symbol('91')) == r"91"
    assert libreOffice(Symbol('alpha_new')) == r"\alpha_{new}"
    assert libreOffice(Symbol('C^orig')) == r"C^{orig}"
    assert libreOffice(Symbol('x^alpha')) == r"x^{\alpha}"
    assert libreOffice(Symbol('beta^alpha')) == r"\beta^{\alpha}"
    assert libreOffice(Symbol('e^Alpha')) == r"e^{A}"
    assert libreOffice(Symbol('omega_alpha^beta')) == r"\omega^{\beta}_{\alpha}"
    assert libreOffice(Symbol('omega') ** Symbol('beta')) == r"\omega^{\beta}"


@XFAIL
def test_latex_symbols_failing():
    rho, mass, volume = symbols('rho, mass, volume')
    assert libreOffice(
        volume * rho == mass) == r"\rho \mathrm{volume} = \mathrm{mass}"
    assert libreOffice(volume / mass * rho == 1) == r"\rho \mathrm{volume} {\mathrm{mass}}^{(-1)} = 1"
    assert libreOffice(mass ** 3 * volume ** 3) == r"{\mathrm{mass}}^{3} \cdot {\mathrm{volume}}^{3}"


def test_latex_functions():
    assert libreOffice(exp(x)) == "e^{x}"
    assert libreOffice(exp(1) + exp(2)) == "e + e^{2}"

    f = Function('f')
    assert libreOffice(f(x)) == r'f{\left (x \right )}'
    assert libreOffice(f) == r'f'

    g = Function('g')
    assert libreOffice(g(x, y)) == r'g{\left (x,y \right )}'
    assert libreOffice(g) == r'g'

    h = Function('h')
    assert libreOffice(h(x, y, z)) == r'h{\left (x,y,z \right )}'
    assert libreOffice(h) == r'h'

    Li = Function('Li')
    assert libreOffice(Li) == r'\operatorname{Li}'
    assert libreOffice(Li(x)) == r'\operatorname{Li}{\left (x \right )}'

    beta = Function('beta')

    # not to be confused with the beta function
    assert libreOffice(beta(x)) == r"\beta{\left (x \right )}"
    assert libreOffice(beta) == r"\beta"

    a1 = Function('a_1')

    assert libreOffice(a1) == r"\operatorname{a_{1}}"
    assert libreOffice(a1(x)) == r"\operatorname{a_{1}}{\left (x \right )}"

    # issue 5868
    omega1 = Function('omega1')
    assert libreOffice(omega1) == r"\omega_{1}"
    assert libreOffice(omega1(x)) == r"\omega_{1}{\left (x \right )}"

    assert libreOffice(sin(x)) == r"\sin{\left (x \right )}"
    assert libreOffice(sin(x), fold_func_brackets=True) == r"\sin {x}"
    assert libreOffice(sin(2 * x ** 2), fold_func_brackets=True) == \
        r"\sin {2 x^{2}}"
    assert libreOffice(sin(x ** 2), fold_func_brackets=True) == \
        r"\sin {x^{2}}"

    assert libreOffice(asin(x) ** 2) == r"\operatorname{asin}^{2}{\left (x \right )}"
    assert libreOffice(asin(x) ** 2, inv_trig_style="full") == \
        r"\arcsin^{2}{\left (x \right )}"
    assert libreOffice(asin(x) ** 2, inv_trig_style="power") == \
        r"\sin^{-1}{\left (x \right )}^{2}"
    assert libreOffice(asin(x ** 2), inv_trig_style="power",
                       fold_func_brackets=True) == \
        r"\sin^{-1} {x^{2}}"

    assert libreOffice(factorial(k)) == r"k!"
    assert libreOffice(factorial(-k)) == r"\left(- k\right)!"

    assert libreOffice(subfactorial(k)) == r"!k"
    assert libreOffice(subfactorial(-k)) == r"!\left(- k\right)"

    assert libreOffice(factorial2(k)) == r"k!!"
    assert libreOffice(factorial2(-k)) == r"\left(- k\right)!!"

    assert libreOffice(binomial(2, k)) == r"{\binom{2}{k}}"

    assert libreOffice(
        FallingFactorial(3, k)) == r"{\left(3\right)}_{\left(k\right)}"
    assert libreOffice(RisingFactorial(3, k)) == r"{\left(3\right)}^{\left(k\right)}"

    assert libreOffice(floor(x)) == r"\lfloor{x}\rfloor"
    assert libreOffice(ceiling(x)) == r"\lceil{x}\rceil"
    assert libreOffice(Min(x, 2, x ** 3)) == r"\min\left(2, x, x^{3}\right)"
    assert libreOffice(Min(x, y) ** 2) == r"\min\left(x, y\right)^{2}"
    assert libreOffice(Max(x, 2, x ** 3)) == r"\max\left(2, x, x^{3}\right)"
    assert libreOffice(Max(x, y) ** 2) == r"\max\left(x, y\right)^{2}"
    assert libreOffice(Abs(x)) == r"\left\lvert{x}\right\rvert"
    assert libreOffice(re(x)) == r"\Re{x}"
    assert libreOffice(re(x + y)) == r"\Re{x} + \Re{y}"
    assert libreOffice(im(x)) == r"\Im{x}"
    assert libreOffice(conjugate(x)) == r"\overline{x}"
    assert libreOffice(gamma(x)) == r"\Gamma{\left(x \right)}"
    w = Wild('w')
    assert libreOffice(gamma(w)) == r"\Gamma{\left(w \right)}"
    assert libreOffice(Order(x)) == r"\mathcal{O}\left(x\right)"
    assert libreOffice(Order(x, x)) == r"\mathcal{O}\left(x\right)"
    assert libreOffice(Order(x, (x, 0))) == r"\mathcal{O}\left(x\right)"
    assert libreOffice(Order(x, (x, oo))) == r"\mathcal{O}\left(x; x\rightarrow\infty\right)"
    assert libreOffice(Order(x, x, y)) == r"\mathcal{O}\left(x; \left ( x, \quad y\right )\rightarrow\left ( 0, \quad 0\right )\right)"
    assert libreOffice(Order(x, x, y)) == r"\mathcal{O}\left(x; \left ( x, \quad y\right )\rightarrow\left ( 0, \quad 0\right )\right)"
    assert libreOffice(Order(x, (x, oo), (y, oo))) == r"\mathcal{O}\left(x; \left ( x, \quad y\right )\rightarrow\left ( \infty, \quad \infty\right )\right)"
    assert libreOffice(lowergamma(x, y)) == r'\gamma\left(x, y\right)'
    assert libreOffice(uppergamma(x, y)) == r'\Gamma\left(x, y\right)'

    assert libreOffice(cot(x)) == r'\cot{\left (x \right )}'
    assert libreOffice(coth(x)) == r'\coth{\left (x \right )}'
    assert libreOffice(re(x)) == r'\Re{x}'
    assert libreOffice(im(x)) == r'\Im{x}'
    assert libreOffice(root(x, y)) == r'x^{\frac{1}{y}}'
    assert libreOffice(arg(x)) == r'\arg{\left (x \right )}'
    assert libreOffice(zeta(x)) == r'\zeta\left(x\right)'

    assert libreOffice(zeta(x)) == r"\zeta\left(x\right)"
    assert libreOffice(zeta(x) ** 2) == r"\zeta^{2}\left(x\right)"
    assert libreOffice(zeta(x, y)) == r"\zeta\left(x, y\right)"
    assert libreOffice(zeta(x, y) ** 2) == r"\zeta^{2}\left(x, y\right)"
    assert libreOffice(dirichlet_eta(x)) == r"\eta\left(x\right)"
    assert libreOffice(dirichlet_eta(x) ** 2) == r"\eta^{2}\left(x\right)"
    assert libreOffice(polylog(x, y)) == r"\operatorname{Li}_{x}\left(y\right)"
    assert libreOffice(
        polylog(x, y)**2) == r"\operatorname{Li}_{x}^{2}\left(y\right)"
    assert libreOffice(lerchphi(x, y, n)) == r"\Phi\left(x, y, n\right)"
    assert libreOffice(lerchphi(x, y, n) ** 2) == r"\Phi^{2}\left(x, y, n\right)"

    assert libreOffice(elliptic_k(z)) == r"K\left(z\right)"
    assert libreOffice(elliptic_k(z) ** 2) == r"K^{2}\left(z\right)"
    assert libreOffice(elliptic_f(x, y)) == r"F\left(x\middle| y\right)"
    assert libreOffice(elliptic_f(x, y) ** 2) == r"F^{2}\left(x\middle| y\right)"
    assert libreOffice(elliptic_e(x, y)) == r"E\left(x\middle| y\right)"
    assert libreOffice(elliptic_e(x, y) ** 2) == r"E^{2}\left(x\middle| y\right)"
    assert libreOffice(elliptic_e(z)) == r"E\left(z\right)"
    assert libreOffice(elliptic_e(z) ** 2) == r"E^{2}\left(z\right)"
    assert libreOffice(elliptic_pi(x, y, z)) == r"\Pi\left(x; y\middle| z\right)"
    assert libreOffice(elliptic_pi(x, y, z) ** 2) == \
        r"\Pi^{2}\left(x; y\middle| z\right)"
    assert libreOffice(elliptic_pi(x, y)) == r"\Pi\left(x\middle| y\right)"
    assert libreOffice(elliptic_pi(x, y) ** 2) == r"\Pi^{2}\left(x\middle| y\right)"

    assert libreOffice(Ei(x)) == r'\operatorname{Ei}{\left (x \right )}'
    assert libreOffice(Ei(x) ** 2) == r'\operatorname{Ei}^{2}{\left (x \right )}'
    assert libreOffice(expint(x, y) ** 2) == r'\operatorname{E}_{x}^{2}\left(y\right)'
    assert libreOffice(Shi(x) ** 2) == r'\operatorname{Shi}^{2}{\left (x \right )}'
    assert libreOffice(Si(x) ** 2) == r'\operatorname{Si}^{2}{\left (x \right )}'
    assert libreOffice(Ci(x) ** 2) == r'\operatorname{Ci}^{2}{\left (x \right )}'
    assert libreOffice(Chi(x) ** 2) == r'\operatorname{Chi}^{2}{\left (x \right )}'
    assert libreOffice(Chi(x)) == r'\operatorname{Chi}{\left (x \right )}'

    assert libreOffice(
        jacobi(n, a, b, x)) == r'P_{n}^{\left(a,b\right)}\left(x\right)'
    assert libreOffice(jacobi(n, a, b, x) ** 2) == r'\left(P_{n}^{\left(a,b\right)}\left(x\right)\right)^{2}'
    assert libreOffice(
        gegenbauer(n, a, x)) == r'C_{n}^{\left(a\right)}\left(x\right)'
    assert libreOffice(gegenbauer(n, a, x) ** 2) == r'\left(C_{n}^{\left(a\right)}\left(x\right)\right)^{2}'
    assert libreOffice(chebyshevt(n, x)) == r'T_{n}\left(x\right)'
    assert libreOffice(
        chebyshevt(n, x)**2) == r'\left(T_{n}\left(x\right)\right)^{2}'
    assert libreOffice(chebyshevu(n, x)) == r'U_{n}\left(x\right)'
    assert libreOffice(
        chebyshevu(n, x)**2) == r'\left(U_{n}\left(x\right)\right)^{2}'
    assert libreOffice(legendre(n, x)) == r'P_{n}\left(x\right)'
    assert libreOffice(legendre(n, x) ** 2) == r'\left(P_{n}\left(x\right)\right)^{2}'
    assert libreOffice(
        assoc_legendre(n, a, x)) == r'P_{n}^{\left(a\right)}\left(x\right)'
    assert libreOffice(assoc_legendre(n, a, x) ** 2) == r'\left(P_{n}^{\left(a\right)}\left(x\right)\right)^{2}'
    assert libreOffice(laguerre(n, x)) == r'L_{n}\left(x\right)'
    assert libreOffice(laguerre(n, x) ** 2) == r'\left(L_{n}\left(x\right)\right)^{2}'
    assert libreOffice(
        assoc_laguerre(n, a, x)) == r'L_{n}^{\left(a\right)}\left(x\right)'
    assert libreOffice(assoc_laguerre(n, a, x) ** 2) == r'\left(L_{n}^{\left(a\right)}\left(x\right)\right)^{2}'
    assert libreOffice(hermite(n, x)) == r'H_{n}\left(x\right)'
    assert libreOffice(hermite(n, x) ** 2) == r'\left(H_{n}\left(x\right)\right)^{2}'

    theta = Symbol("theta", real=True)
    phi = Symbol("phi", real=True)
    assert libreOffice(Ynm(n, m, theta, phi)) == r'Y_{n}^{m}\left(\theta,\phi\right)'
    assert libreOffice(Ynm(n, m, theta, phi) ** 3) == r'\left(Y_{n}^{m}\left(\theta,\phi\right)\right)^{3}'
    assert libreOffice(Znm(n, m, theta, phi)) == r'Z_{n}^{m}\left(\theta,\phi\right)'
    assert libreOffice(Znm(n, m, theta, phi) ** 3) == r'\left(Z_{n}^{m}\left(\theta,\phi\right)\right)^{3}'

    # Test latex printing of function names with "_"
    assert libreOffice(
        polar_lift(0)) == r"\operatorname{polar\_lift}{\left (0 \right )}"
    assert libreOffice(polar_lift(
        0) ** 3) == r"\operatorname{polar\_lift}^{3}{\left (0 \right )}"

    assert libreOffice(totient(n)) == r'\phi\left( n \right)'

    assert libreOffice(divisor_sigma(x)) == r"\sigma\left(x\right)"
    assert libreOffice(divisor_sigma(x) ** 2) == r"\sigma^{2}\left(x\right)"
    assert libreOffice(divisor_sigma(x, y)) == r"\sigma_y\left(x\right)"
    assert libreOffice(divisor_sigma(x, y) ** 2) == r"\sigma^{2}_y\left(x\right)"

    # some unknown function name should get rendered with \operatorname
    fjlkd = Function('fjlkd')
    assert libreOffice(fjlkd(x)) == r'\operatorname{fjlkd}{\left (x \right )}'
    # even when it is referred to without an argument
    assert libreOffice(fjlkd) == r'\operatorname{fjlkd}'

def test_hyper_printing():
    from sympy import pi

    assert libreOffice(meijerg(Tuple(pi, pi, x), Tuple(1),
                               (0, 1), Tuple(1, 2, 3/pi), z)) == \
        r'{G_{4, 5}^{2, 3}\left(\begin{matrix} \pi, \pi, x & 1 \\0, 1 & 1, 2, \frac{3}{\pi} \end{matrix} \middle| {z} \right)}'
    assert libreOffice(meijerg(Tuple(), Tuple(1), (0,), Tuple(), z)) == \
        r'{G_{1, 1}^{1, 0}\left(\begin{matrix}  & 1 \\0 &  \end{matrix} \middle| {z} \right)}'
    assert libreOffice(hyper((x, 2), (3,), z)) == \
        r'{{}_{2}F_{1}\left(\begin{matrix} x, 2 ' \
        r'\\ 3 \end{matrix}\middle| {z} \right)}'
    assert libreOffice(hyper(Tuple(), Tuple(1), z)) == \
        r'{{}_{0}F_{1}\left(\begin{matrix}  ' \
        r'\\ 1 \end{matrix}\middle| {z} \right)}'


def test_latex_bessel():
    from sympy.functions.special.bessel import (besselj, bessely, besseli,
            besselk, hankel1, hankel2, jn, yn)

    assert libreOffice(besselj(n, z ** 2) ** k) == r'J^{k}_{n}\left(z^{2}\right)'
    assert libreOffice(bessely(n, z)) == r'Y_{n}\left(z\right)'
    assert libreOffice(besseli(n, z)) == r'I_{n}\left(z\right)'
    assert libreOffice(besselk(n, z)) == r'K_{n}\left(z\right)'
    assert libreOffice(hankel1(n, z ** 2) ** 2) == \
        r'\left(H^{(1)}_{n}\left(z^{2}\right)\right)^{2}'
    assert libreOffice(hankel2(n, z)) == r'H^{(2)}_{n}\left(z\right)'
    assert libreOffice(jn(n, z)) == r'j_{n}\left(z\right)'
    assert libreOffice(yn(n, z)) == r'y_{n}\left(z\right)'


def test_latex_fresnel():
    from sympy.functions.special.error_functions import (fresnels, fresnelc)
    assert libreOffice(fresnels(z)) == r'S\left(z\right)'
    assert libreOffice(fresnelc(z)) == r'C\left(z\right)'
    assert libreOffice(fresnels(z) ** 2) == r'S^{2}\left(z\right)'
    assert libreOffice(fresnelc(z) ** 2) == r'C^{2}\left(z\right)'


def test_latex_brackets():
    assert libreOffice((-1) ** x) == r"\left(-1\right)^{x}"


def test_latex_indexed():
    Psi_symbol = Symbol('Psi_0', complex=True, real=False)
    Psi_indexed = IndexedBase(Symbol('Psi', complex=True, real=False))
    symbol_latex = libreOffice(Psi_symbol * conjugate(Psi_symbol))
    indexed_latex = libreOffice(Psi_indexed[0] * conjugate(Psi_indexed[0]))
    # \\overline{\\Psi_{0}} \\Psi_{0}   vs.   \\Psi_{0} \\overline{\\Psi_{0}}
    assert symbol_latex.split() == indexed_latex.split() \
        or symbol_latex.split() == indexed_latex.split()[::-1]

    # Symbol('gamma') gives r'\gamma'
    assert libreOffice(IndexedBase('gamma')) == r'\gamma'
    assert libreOffice(IndexedBase('a b')) == 'a b'
    assert libreOffice(IndexedBase('a_b')) == 'a_{b}'


def test_latex_derivatives():
    # regular "d" for ordinary derivatives
    assert libreOffice(diff(x ** 3, x, evaluate=False)) == \
        r"\frac{d}{d x} x^{3}"
    assert libreOffice(diff(sin(x) + x ** 2, x, evaluate=False)) == \
        r"\frac{d}{d x}\left(x^{2} + \sin{\left (x \right )}\right)"
    assert libreOffice(diff(diff(sin(x) + x ** 2, x, evaluate=False), evaluate=False)) == \
        r"\frac{d^{2}}{d x^{2}} \left(x^{2} + \sin{\left (x \right )}\right)"
    assert libreOffice(diff(diff(diff(sin(x) + x ** 2, x, evaluate=False), evaluate=False), evaluate=False)) == \
        r"\frac{d^{3}}{d x^{3}} \left(x^{2} + \sin{\left (x \right )}\right)"

    # \partial for partial derivatives
    assert libreOffice(diff(sin(x * y), x, evaluate=False)) == \
        r"\frac{\partial}{\partial x} \sin{\left (x y \right )}"
    assert libreOffice(diff(sin(x * y) + x ** 2, x, evaluate=False)) == \
        r"\frac{\partial}{\partial x}\left(x^{2} + \sin{\left (x y \right )}\right)"
    assert libreOffice(diff(diff(sin(x * y) + x ** 2, x, evaluate=False), x, evaluate=False)) == \
        r"\frac{\partial^{2}}{\partial x^{2}} \left(x^{2} + \sin{\left (x y \right )}\right)"
    assert libreOffice(diff(diff(diff(sin(x * y) + x ** 2, x, evaluate=False), x, evaluate=False), x, evaluate=False)) == \
        r"\frac{\partial^{3}}{\partial x^{3}} \left(x^{2} + \sin{\left (x y \right )}\right)"

    # mixed partial derivatives
    f = Function("f")
    assert libreOffice(diff(diff(f(x, y), x, evaluate=False), y, evaluate=False)) == \
        r"\frac{\partial^{2}}{\partial x\partial y}  " + libreOffice(f(x, y))

    assert libreOffice(diff(diff(diff(f(x, y), x, evaluate=False), x, evaluate=False), y, evaluate=False)) == \
        r"\frac{\partial^{3}}{\partial x^{2}\partial y}  " + libreOffice(f(x, y))

    # use ordinary d when one of the variables has been integrated out
    assert libreOffice(diff(Integral(exp(-x * y), (x, 0, oo)), y, evaluate=False)) == \
        r"\frac{d}{d y} \int_{0}^{\infty} e^{- x y}\, dx"

def test_latex_subs():
    assert libreOffice(Subs(x * y, (
        x, y), (1, 2))) == r'\left. x y \right|_{\substack{ x=1\\ y=2 }}'


def test_latex_integrals():
    assert libreOffice(Integral(log(x), x)) == r"\int \log{\left (x \right )}\, dx"
    assert libreOffice(Integral(x ** 2, (x, 0, 1))) == r"\int_{0}^{1} x^{2}\, dx"
    assert libreOffice(Integral(x ** 2, (x, 10, 20))) == r"\int_{10}^{20} x^{2}\, dx"
    assert libreOffice(Integral(
        y*x**2, (x, 0, 1), y)) == r"\int\int_{0}^{1} x^{2} y\, dx\, dy"
    assert libreOffice(Integral(y * x ** 2, (x, 0, 1), y), mode='equation*') \
           == r"\begin{equation*}\int\int\limits_{0}^{1} x^{2} y\, dx\, dy\end{equation*}"
    assert libreOffice(Integral(y * x ** 2, (x, 0, 1), y), mode='equation*', itex=True) \
           == r"$$\int\int_{0}^{1} x^{2} y\, dx\, dy$$"
    assert libreOffice(Integral(x, (x, 0))) == r"\int^{0} x\, dx"
    assert libreOffice(Integral(x * y, x, y)) == r"\iint x y\, dx\, dy"
    assert libreOffice(Integral(x * y * z, x, y, z)) == r"\iiint x y z\, dx\, dy\, dz"
    assert libreOffice(Integral(x * y * z * t, x, y, z, t)) == \
        r"\iiiint t x y z\, dx\, dy\, dz\, dt"
    assert libreOffice(Integral(x, x, x, x, x, x, x)) == \
        r"\int\int\int\int\int\int x\, dx\, dx\, dx\, dx\, dx\, dx"
    assert libreOffice(Integral(x, x, y, (z, 0, 1))) == \
        r"\int_{0}^{1}\int\int x\, dx\, dy\, dz"


def test_latex_sets():
    for s in (frozenset, set):
        assert libreOffice(s([x * y, x ** 2])) == r"\left\{x^{2}, x y\right\}"
        assert libreOffice(s(range(1, 6))) == r"\left\{1, 2, 3, 4, 5\right\}"
        assert libreOffice(s(range(1, 13))) == \
            r"\left\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\right\}"

    s = FiniteSet
    assert libreOffice(s(*[x * y, x ** 2])) == r"\left\{x^{2}, x y\right\}"
    assert libreOffice(s(*range(1, 6))) == r"\left\{1, 2, 3, 4, 5\right\}"
    assert libreOffice(s(*range(1, 13))) == \
        r"\left\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\right\}"


def test_latex_Range():
    assert libreOffice(Range(1, 51)) == \
        r'\left\{1, 2, \ldots, 50\right\}'
    assert libreOffice(Range(1, 4)) == r'\left\{1, 2, 3\right\}'


def test_latex_intervals():
    a = Symbol('a', real=True)
    assert libreOffice(Interval(0, 0)) == r"\left\{0\right\}"
    assert libreOffice(Interval(0, a)) == r"\left[0, a\right]"
    assert libreOffice(Interval(0, a, False, False)) == r"\left[0, a\right]"
    assert libreOffice(Interval(0, a, True, False)) == r"\left(0, a\right]"
    assert libreOffice(Interval(0, a, False, True)) == r"\left[0, a\right)"
    assert libreOffice(Interval(0, a, True, True)) == r"\left(0, a\right)"


def test_latex_emptyset():
    assert libreOffice(S.EmptySet) == r"\emptyset"


def test_latex_union():
    assert libreOffice(Union(Interval(0, 1), Interval(2, 3))) == \
        r"\left[0, 1\right] \cup \left[2, 3\right]"
    assert libreOffice(Union(Interval(1, 1), Interval(2, 2), Interval(3, 4))) == \
        r"\left\{1, 2\right\} \cup \left[3, 4\right]"


def test_latex_Complement():
    assert libreOffice(Complement(S.Reals, S.Naturals)) == r"\mathbb{R} \setminus \mathbb{N}"


def test_latex_productset():
    line = Interval(0, 1)
    bigline = Interval(0, 10)
    fset = FiniteSet(1, 2, 3)
    assert libreOffice(line ** 2) == r"%s^2" % libreOffice(line)
    assert libreOffice(line * bigline * fset) == r"%s \times %s \times %s" % (
        libreOffice(line), libreOffice(bigline), libreOffice(fset))


def test_latex_Naturals():
    assert libreOffice(S.Naturals) == r"\mathbb{N}"
    assert libreOffice(S.Integers) == r"\mathbb{Z}"


def test_latex_ImageSet():
    x = Symbol('x')
    assert libreOffice(ImageSet(Lambda(x, x ** 2), S.Naturals)) == \
        r"\left\{x^{2}\; |\; x \in \mathbb{N}\right\}"


def test_latex_Contains():
    x = Symbol('x')
    assert libreOffice(Contains(x, S.Naturals)) == r"x \in \mathbb{N}"


def test_latex_sum():
    assert libreOffice(Sum(x * y ** 2, (x, -2, 2), (y, -5, 5))) == \
        r"\sum_{\substack{-2 \leq x \leq 2\\-5 \leq y \leq 5}} x y^{2}"
    assert libreOffice(Sum(x ** 2, (x, -2, 2))) == \
        r"\sum_{x=-2}^{2} x^{2}"
    assert libreOffice(Sum(x ** 2 + y, (x, -2, 2))) == \
        r"\sum_{x=-2}^{2} \left(x^{2} + y\right)"


def test_latex_product():
    assert libreOffice(Product(x * y ** 2, (x, -2, 2), (y, -5, 5))) == \
        r"\prod_{\substack{-2 \leq x \leq 2\\-5 \leq y \leq 5}} x y^{2}"
    assert libreOffice(Product(x ** 2, (x, -2, 2))) == \
        r"\prod_{x=-2}^{2} x^{2}"
    assert libreOffice(Product(x ** 2 + y, (x, -2, 2))) == \
        r"\prod_{x=-2}^{2} \left(x^{2} + y\right)"


def test_latex_limits():
    assert libreOffice(Limit(x, x, oo)) == r"\lim_{x \to \infty} x"

    # issue 8175
    f = Function('f')
    assert libreOffice(Limit(f(x), x, 0)) == r"\lim_{x \to 0^+} f{\left (x \right )}"
    assert libreOffice(Limit(f(x), x, 0, "-")) == r"\lim_{x \to 0^-} f{\left (x \right )}"


def test_issue_3568():
    beta = Symbol(r'\beta')
    y = beta + x
    assert libreOffice(y) in [r'\beta + x', r'x + \beta']

    beta = Symbol(r'beta')
    y = beta + x
    assert libreOffice(y) in [r'\beta + x', r'x + \beta']


def test_latex():
    assert libreOffice((2 * tau) ** Rational(7, 2)) == "8 \\sqrt{2} \\tau^{\\frac{7}{2}}"
    assert libreOffice((2 * mu) ** Rational(7, 2), mode='equation*') == \
        "\\begin{equation*}8 \\sqrt{2} \\mu^{\\frac{7}{2}}\\end{equation*}"
    assert libreOffice((2 * mu) ** Rational(7, 2), mode='equation', itex=True) == \
        "$$8 \\sqrt{2} \\mu^{\\frac{7}{2}}$$"
    assert libreOffice([2 / x, y]) == r"\left [ \frac{2}{x}, \quad y\right ]"


def test_latex_dict():
    d = {Rational(1): 1, x**2: 2, x: 3, x**3: 4}
    assert libreOffice(d) == r'\left \{ 1 : 1, \quad x : 3, \quad x^{2} : 2, \quad x^{3} : 4\right \}'
    D = Dict(d)
    assert libreOffice(D) == r'\left \{ 1 : 1, \quad x : 3, \quad x^{2} : 2, \quad x^{3} : 4\right \}'


def test_latex_list():
    l = [Symbol('omega1'), Symbol('a'), Symbol('alpha')]
    assert libreOffice(l) == r'\left [ \omega_{1}, \quad a, \quad \alpha\right ]'


def test_latex_rational():
    #tests issue 3973
    assert libreOffice(-Rational(1, 2)) == "- \\frac{1}{2}"
    assert libreOffice(Rational(-1, 2)) == "- \\frac{1}{2}"
    assert libreOffice(Rational(1, -2)) == "- \\frac{1}{2}"
    assert libreOffice(-Rational(-1, 2)) == "\\frac{1}{2}"
    assert libreOffice(-Rational(1, 2) * x) == "- \\frac{x}{2}"
    assert libreOffice(-Rational(1, 2) * x + Rational(-2, 3) * y) == \
        "- \\frac{x}{2} - \\frac{2 y}{3}"


def test_latex_inverse():
    #tests issue 4129
    assert libreOffice(1 / x) == "\\frac{1}{x}"
    assert libreOffice(1 / (x + y)) == "\\frac{1}{x + y}"


def test_latex_DiracDelta():
    assert libreOffice(DiracDelta(x)) == r"\delta\left(x\right)"
    assert libreOffice(DiracDelta(x) ** 2) == r"\left(\delta\left(x\right)\right)^{2}"
    assert libreOffice(DiracDelta(x, 0)) == r"\delta\left(x\right)"
    assert libreOffice(DiracDelta(x, 5)) == \
        r"\delta^{\left( 5 \right)}\left( x \right)"
    assert libreOffice(DiracDelta(x, 5) ** 2) == \
        r"\left(\delta^{\left( 5 \right)}\left( x \right)\right)^{2}"


def test_latex_Heaviside():
    assert libreOffice(Heaviside(x)) == r"\theta\left(x\right)"
    assert libreOffice(Heaviside(x) ** 2) == r"\left(\theta\left(x\right)\right)^{2}"


def test_latex_KroneckerDelta():
    assert libreOffice(KroneckerDelta(x, y)) == r"\delta_{x y}"
    assert libreOffice(KroneckerDelta(x, y + 1)) == r"\delta_{x, y + 1}"
    # issue 6578
    assert libreOffice(KroneckerDelta(x + 1, y)) == r"\delta_{y, x + 1}"


def test_latex_LeviCivita():
    assert libreOffice(LeviCivita(x, y, z)) == r"\varepsilon_{x y z}"
    assert libreOffice(LeviCivita(x, y, z) ** 2) == r"\left(\varepsilon_{x y z}\right)^{2}"
    assert libreOffice(LeviCivita(x, y, z + 1)) == r"\varepsilon_{x, y, z + 1}"
    assert libreOffice(LeviCivita(x, y + 1, z)) == r"\varepsilon_{x, y + 1, z}"
    assert libreOffice(LeviCivita(x + 1, y, z)) == r"\varepsilon_{x + 1, y, z}"


def test_mode():
    expr = x + y
    assert libreOffice(expr) == 'x + y'
    assert libreOffice(expr, mode='plain') == 'x + y'
    assert libreOffice(expr, mode='inline') == '$x + y$'
    assert libreOffice(
        expr, mode='equation*') == '\\begin{equation*}x + y\\end{equation*}'
    assert libreOffice(
        expr, mode='equation') == '\\begin{equation}x + y\\end{equation}'


def test_latex_Piecewise():
    p = Piecewise((x, x < 1), (x**2, True))
    assert libreOffice(p) == "\\begin{cases} x & \\text{for}\: x < 1 \\\\x^{2} &" \
                       " \\text{otherwise} \\end{cases}"
    assert libreOffice(p, itex=True) == "\\begin{cases} x & \\text{for}\: x \\lt 1 \\\\x^{2} &" \
                                  " \\text{otherwise} \\end{cases}"
    p = Piecewise((x, x < 0), (0, x >= 0))
    assert libreOffice(p) == "\\begin{cases} x & \\text{for}\\: x < 0 \\\\0 &" \
                       " \\text{for}\\: x \\geq 0 \\end{cases}"
    A, B = symbols("A B", commutative=False)
    p = Piecewise((A**2, Eq(A, B)), (A*B, True))
    s = r"\begin{cases} A^{2} & \text{for}\: A = B \\A B & \text{otherwise} \end{cases}"
    assert libreOffice(p) == s
    assert libreOffice(A * p) == r"A %s" % s
    assert libreOffice(p * A) == r"\left(%s\right) A" % s


def test_latex_Matrix():
    M = Matrix([[1 + x, y], [y, x - 1]])
    assert libreOffice(M) == \
        r'\left[\begin{matrix}x + 1 & y\\y & x - 1\end{matrix}\right]'
    assert libreOffice(M, mode='inline') == \
        r'$\left[\begin{smallmatrix}x + 1 & y\\' \
        r'y & x - 1\end{smallmatrix}\right]$'
    assert libreOffice(M, mat_str='array') == \
        r'\left[\begin{array}{cc}x + 1 & y\\y & x - 1\end{array}\right]'
    assert libreOffice(M, mat_str='bmatrix') == \
        r'\left[\begin{bmatrix}x + 1 & y\\y & x - 1\end{bmatrix}\right]'
    assert libreOffice(M, mat_delim=None, mat_str='bmatrix') == \
        r'\begin{bmatrix}x + 1 & y\\y & x - 1\end{bmatrix}'
    M2 = Matrix(1, 11, range(11))
    assert libreOffice(M2) == \
        r'\left[\begin{array}{ccccccccccc}' \
        r'0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\end{array}\right]'


def test_latex_matrix_with_functions():
    t = symbols('t')
    theta1 = symbols('theta1', cls=Function)

    M = Matrix([[sin(theta1(t)), cos(theta1(t))],
                [cos(theta1(t).diff(t)), sin(theta1(t).diff(t))]])

    expected = (r'\left[\begin{matrix}\sin{\left '
                r'(\theta_{1}{\left (t \right )} \right )} & '
                r'\cos{\left (\theta_{1}{\left (t \right )} \right '
                r')}\\\cos{\left (\frac{d}{d t} \theta_{1}{\left (t '
                r'\right )} \right )} & \sin{\left (\frac{d}{d t} '
                r'\theta_{1}{\left (t \right )} \right '
                r')}\end{matrix}\right]')

    assert libreOffice(M) == expected


def test_latex_mul_symbol():
    assert libreOffice(4 * 4 ** x, mul_symbol='times') == "4 \\times 4^{x}"
    assert libreOffice(4 * 4 ** x, mul_symbol='dot') == "4 \\cdot 4^{x}"
    assert libreOffice(4 * 4 ** x, mul_symbol='ldot') == "4 \,.\, 4^{x}"

    assert libreOffice(4 * x, mul_symbol='times') == "4 \\times x"
    assert libreOffice(4 * x, mul_symbol='dot') == "4 \\cdot x"
    assert libreOffice(4 * x, mul_symbol='ldot') == "4 \,.\, x"


def test_latex_issue_4381():
    y = 4*4**log(2)
    assert libreOffice(y) == r'4 \cdot 4^{\log{\left (2 \right )}}'
    assert libreOffice(1 / y) == r'\frac{1}{4 \cdot 4^{\log{\left (2 \right )}}}'


def test_latex_issue_4576():
    assert libreOffice(Symbol("beta_13_2")) == r"\beta_{13 2}"
    assert libreOffice(Symbol("beta_132_20")) == r"\beta_{132 20}"
    assert libreOffice(Symbol("beta_13")) == r"\beta_{13}"
    assert libreOffice(Symbol("x_a_b")) == r"x_{a b}"
    assert libreOffice(Symbol("x_1_2_3")) == r"x_{1 2 3}"
    assert libreOffice(Symbol("x_a_b1")) == r"x_{a b1}"
    assert libreOffice(Symbol("x_a_1")) == r"x_{a 1}"
    assert libreOffice(Symbol("x_1_a")) == r"x_{1 a}"
    assert libreOffice(Symbol("x_1^aa")) == r"x^{aa}_{1}"
    assert libreOffice(Symbol("x_1__aa")) == r"x^{aa}_{1}"
    assert libreOffice(Symbol("x_11^a")) == r"x^{a}_{11}"
    assert libreOffice(Symbol("x_11__a")) == r"x^{a}_{11}"
    assert libreOffice(Symbol("x_a_a_a_a")) == r"x_{a a a a}"
    assert libreOffice(Symbol("x_a_a^a^a")) == r"x^{a a}_{a a}"
    assert libreOffice(Symbol("x_a_a__a__a")) == r"x^{a a}_{a a}"
    assert libreOffice(Symbol("alpha_11")) == r"\alpha_{11}"
    assert libreOffice(Symbol("alpha_11_11")) == r"\alpha_{11 11}"
    assert libreOffice(Symbol("alpha_alpha")) == r"\alpha_{\alpha}"
    assert libreOffice(Symbol("alpha^aleph")) == r"\alpha^{\aleph}"
    assert libreOffice(Symbol("alpha__aleph")) == r"\alpha^{\aleph}"


def test_latex_pow_fraction():
    x = Symbol('x')
    # Testing exp
    assert 'e^{-x}' in libreOffice(exp(-x) / 2).replace(' ', '')  # Remove Whitespace

    # Testing just e^{-x} in case future changes alter behavior of muls or fracs
    # In particular current output is \frac{1}{2}e^{- x} but perhaps this will
    # change to \frac{e^{-x}}{2}

    # Testing general, non-exp, power
    assert '3^{-x}' in libreOffice(3 ** -x / 2).replace(' ', '')


def test_noncommutative():
    A, B, C = symbols('A,B,C', commutative=False)

    assert libreOffice(A * B * C ** -1) == "A B C^{-1}"
    assert libreOffice(C ** -1 * A * B) == "C^{-1} A B"
    assert libreOffice(A * C ** -1 * B) == "A C^{-1} B"


def test_latex_order():
    expr = x**3 + x**2*y + 3*x*y**3 + y**4

    assert libreOffice(expr, order='lex') == "x^{3} + x^{2} y + 3 x y^{3} + y^{4}"
    assert libreOffice(
        expr, order='rev-lex') == "y^{4} + 3 x y^{3} + x^{2} y + x^{3}"


def test_latex_Lambda():
    assert libreOffice(Lambda(x, x + 1)) == \
        r"\left( x \mapsto x + 1 \right)"
    assert libreOffice(Lambda((x, y), x + 1)) == \
        r"\left( \left ( x, \quad y\right ) \mapsto x + 1 \right)"


def test_latex_PolyElement():
    Ruv, u,v = ring("u,v", ZZ)
    Rxyz, x,y,z = ring("x,y,z", Ruv)

    assert libreOffice(x - x) == r"0"
    assert libreOffice(x - 1) == r"x - 1"
    assert libreOffice(x + 1) == r"x + 1"

    assert libreOffice((u ** 2 + 3 * u * v + 1) * x ** 2 * y + u + 1) == r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + u + 1"
    assert libreOffice((u ** 2 + 3 * u * v + 1) * x ** 2 * y + (u + 1) * x) == r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + \left(u + 1\right) x"
    assert libreOffice((u ** 2 + 3 * u * v + 1) * x ** 2 * y + (u + 1) * x + 1) == r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + \left(u + 1\right) x + 1"
    assert libreOffice((-u ** 2 + 3 * u * v - 1) * x ** 2 * y - (u + 1) * x - 1) == r"-\left({u}^{2} - 3 u v + 1\right) {x}^{2} y - \left(u + 1\right) x - 1"

    assert libreOffice(-(v ** 2 + v + 1) * x + 3 * u * v + 1) == r"-\left({v}^{2} + v + 1\right) x + 3 u v + 1"
    assert libreOffice(-(v ** 2 + v + 1) * x - 3 * u * v + 1) == r"-\left({v}^{2} + v + 1\right) x - 3 u v + 1"


def test_latex_FracElement():
    Fuv, u,v = field("u,v", ZZ)
    Fxyzt, x,y,z,t = field("x,y,z,t", Fuv)

    assert libreOffice(x - x) == r"0"
    assert libreOffice(x - 1) == r"x - 1"
    assert libreOffice(x + 1) == r"x + 1"

    assert libreOffice(x / 3) == r"\frac{x}{3}"
    assert libreOffice(x / z) == r"\frac{x}{z}"
    assert libreOffice(x * y / z) == r"\frac{x y}{z}"
    assert libreOffice(x / (z * t)) == r"\frac{x}{z t}"
    assert libreOffice(x * y / (z * t)) == r"\frac{x y}{z t}"

    assert libreOffice((x - 1) / y) == r"\frac{x - 1}{y}"
    assert libreOffice((x + 1) / y) == r"\frac{x + 1}{y}"
    assert libreOffice((-x - 1) / y) == r"\frac{-x - 1}{y}"
    assert libreOffice((x + 1) / (y * z)) == r"\frac{x + 1}{y z}"
    assert libreOffice(-y / (x + 1)) == r"\frac{-y}{x + 1}"
    assert libreOffice(y * z / (x + 1)) == r"\frac{y z}{x + 1}"

    assert libreOffice(((u + 1) * x * y + 1) / ((v - 1) * z - 1)) == r"\frac{\left(u + 1\right) x y + 1}{\left(v - 1\right) z - 1}"
    assert libreOffice(((u + 1) * x * y + 1) / ((v - 1) * z - t * u * v - 1)) == r"\frac{\left(u + 1\right) x y + 1}{\left(v - 1\right) z - u v t - 1}"


def test_latex_Poly():
    assert libreOffice(Poly(x ** 2 + 2 * x, x)) == \
        r"\operatorname{Poly}{\left( x^{2} + 2 x, x, domain=\mathbb{Z} \right)}"
    assert libreOffice(Poly(x / y, x)) == \
        r"\operatorname{Poly}{\left( \frac{x}{y}, x, domain=\mathbb{Z}\left(y\right) \right)}"
    assert libreOffice(Poly(2.0 * x + y)) == \
        r"\operatorname{Poly}{\left( 2.0 x + 1.0 y, x, y, domain=\mathbb{R} \right)}"


def test_latex_RootOf():
    assert libreOffice(RootOf(x ** 5 + x + 3, 0)) == \
        r"\operatorname{RootOf} {\left(x^{5} + x + 3, 0\right)}"


def test_latex_RootSum():
    assert libreOffice(RootSum(x ** 5 + x + 3, sin)) == \
        r"\operatorname{RootSum} {\left(x^{5} + x + 3, \left( x \mapsto \sin{\left (x \right )} \right)\right)}"


def test_settings():
    raises(TypeError, lambda: libreOffice(x * y, method="garbage"))


def test_latex_numbers():
    assert libreOffice(catalan(n)) == r"C_{n}"


def test_lamda():
    assert libreOffice(Symbol('lamda')) == r"\lambda"
    assert libreOffice(Symbol('Lamda')) == r"\Lambda"


def test_custom_symbol_names():
    x = Symbol('x')
    y = Symbol('y')
    assert libreOffice(x) == "x"
    assert libreOffice(x, symbol_names={x: "x_i"}) == "x_i"
    assert libreOffice(x + y, symbol_names={x: "x_i"}) == "x_i + y"
    assert libreOffice(x ** 2, symbol_names={x: "x_i"}) == "x_i^{2}"
    assert libreOffice(x + y, symbol_names={x: "x_i", y: "y_j"}) == "x_i + y_j"


def test_matAdd():
    from sympy import MatrixSymbol
    from sympy.printing.latex import LatexPrinter
    C = MatrixSymbol('C', 5, 5)
    B = MatrixSymbol('B', 5, 5)
    l = LatexPrinter()
    assert l._print_MatAdd(C - 2*B) in ['-2 B + C', 'C -2 B']
    assert l._print_MatAdd(C + 2*B) in ['2 B + C', 'C + 2 B']
    assert l._print_MatAdd(B - 2*C) in ['B -2 C', '-2 C + B']
    assert l._print_MatAdd(B + 2*C) in ['B + 2 C', '2 C + B']


def test_matMul():
    from sympy import MatrixSymbol
    from sympy.printing.latex import LatexPrinter
    A = MatrixSymbol('A', 5, 5)
    B = MatrixSymbol('B', 5, 5)
    x = Symbol('x')
    l = LatexPrinter()
    assert l._print_MatMul(2*A) == '2 A'
    assert l._print_MatMul(2*x*A) == '2 x A'
    assert l._print_MatMul(-2*A) == '-2 A'
    assert l._print_MatMul(1.5*A) == '1.5 A'
    assert l._print_MatMul(sqrt(2)*A) == r'\sqrt{2} A'
    assert l._print_MatMul(-sqrt(2)*A) == r'- \sqrt{2} A'
    assert l._print_MatMul(2*sqrt(2)*x*A) == r'2 \sqrt{2} x A'
    assert l._print_MatMul(-2*A*(A + 2*B)) in [r'-2 A \left(A + 2 B\right)',
        r'-2 A \left(2 B + A\right)']

def test_latex_MatrixSlice():
    from sympy.matrices.expressions import MatrixSymbol
    assert libreOffice(MatrixSymbol('X', 10, 10)[:5, 1:9:2]) == \
            r'X\left[:5, 1:9:2\right]'
    assert libreOffice(MatrixSymbol('X', 10, 10)[5, :5:2]) == \
            r'X\left[5, :5:2\right]'

def test_latex_RandomDomain():
    from sympy.stats import Normal, Die, Exponential, pspace, where
    X = Normal('x1', 0, 1)
    assert libreOffice(where(X > 0)) == r"Domain: 0 < x_{1} \wedge x_{1} < \infty"

    D = Die('d1', 6)
    assert libreOffice(where(D > 4)) == r"Domain: d_{1} = 5 \vee d_{1} = 6"

    A = Exponential('a', 1)
    B = Exponential('b', 1)
    assert libreOffice(
        pspace(Tuple(A, B)).domain) == \
        r"Domain: 0 \leq a \wedge 0 \leq b \wedge a < \infty \wedge b < \infty"


def test_PrettyPoly():
    from sympy.polys.domains import QQ
    F = QQ.frac_field(x, y)
    R = QQ[x, y]

    assert libreOffice(F.convert(x / (x + y))) == libreOffice(x / (x + y))
    assert libreOffice(R.convert(x + y)) == libreOffice(x + y)


def test_integral_transforms():
    x = Symbol("x")
    k = Symbol("k")
    f = Function("f")
    a = Symbol("a")
    b = Symbol("b")

    assert libreOffice(MellinTransform(f(x), x, k)) == r"\mathcal{M}_{x}\left[f{\left (x \right )}\right]\left(k\right)"
    assert libreOffice(InverseMellinTransform(f(k), k, x, a, b)) == r"\mathcal{M}^{-1}_{k}\left[f{\left (k \right )}\right]\left(x\right)"

    assert libreOffice(LaplaceTransform(f(x), x, k)) == r"\mathcal{L}_{x}\left[f{\left (x \right )}\right]\left(k\right)"
    assert libreOffice(InverseLaplaceTransform(f(k), k, x, (a, b))) == r"\mathcal{L}^{-1}_{k}\left[f{\left (k \right )}\right]\left(x\right)"

    assert libreOffice(FourierTransform(f(x), x, k)) == r"\mathcal{F}_{x}\left[f{\left (x \right )}\right]\left(k\right)"
    assert libreOffice(InverseFourierTransform(f(k), k, x)) == r"\mathcal{F}^{-1}_{k}\left[f{\left (k \right )}\right]\left(x\right)"

    assert libreOffice(CosineTransform(f(x), x, k)) == r"\mathcal{COS}_{x}\left[f{\left (x \right )}\right]\left(k\right)"
    assert libreOffice(InverseCosineTransform(f(k), k, x)) == r"\mathcal{COS}^{-1}_{k}\left[f{\left (k \right )}\right]\left(x\right)"

    assert libreOffice(SineTransform(f(x), x, k)) == r"\mathcal{SIN}_{x}\left[f{\left (x \right )}\right]\left(k\right)"
    assert libreOffice(InverseSineTransform(f(k), k, x)) == r"\mathcal{SIN}^{-1}_{k}\left[f{\left (k \right )}\right]\left(x\right)"


def test_PolynomialRingBase():
    from sympy.polys.domains import QQ
    assert libreOffice(QQ.old_poly_ring(x, y)) == r"\mathbb{Q}\left[x, y\right]"
    assert libreOffice(QQ.old_poly_ring(x, y, order="ilex")) == \
        r"S_<^{-1}\mathbb{Q}\left[x, y\right]"


def test_categories():
    from sympy.categories import (Object, IdentityMorphism,
        NamedMorphism, Category, Diagram, DiagramGrid)

    A1 = Object("A1")
    A2 = Object("A2")
    A3 = Object("A3")

    f1 = NamedMorphism(A1, A2, "f1")
    f2 = NamedMorphism(A2, A3, "f2")
    id_A1 = IdentityMorphism(A1)

    K1 = Category("K1")

    assert libreOffice(A1) == "A_{1}"
    assert libreOffice(f1) == "f_{1}:A_{1}\\rightarrow A_{2}"
    assert libreOffice(id_A1) == "id:A_{1}\\rightarrow A_{1}"
    assert libreOffice(f2 * f1) == "f_{2}\\circ f_{1}:A_{1}\\rightarrow A_{3}"

    assert libreOffice(K1) == "\mathbf{K_{1}}"

    d = Diagram()
    assert libreOffice(d) == "\emptyset"

    d = Diagram({f1: "unique", f2: S.EmptySet})
    assert libreOffice(d) == r"\left \{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \emptyset, \quad id:A_{1}\rightarrow " \
        r"A_{1} : \emptyset, \quad id:A_{2}\rightarrow A_{2} : " \
        r"\emptyset, \quad id:A_{3}\rightarrow A_{3} : \emptyset, " \
        r"\quad f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}, " \
        r"\quad f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right \}"

    d = Diagram({f1: "unique", f2: S.EmptySet}, {f2 * f1: "unique"})
    assert libreOffice(d) == r"\left \{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \emptyset, \quad id:A_{1}\rightarrow " \
        r"A_{1} : \emptyset, \quad id:A_{2}\rightarrow A_{2} : " \
        r"\emptyset, \quad id:A_{3}\rightarrow A_{3} : \emptyset, " \
        r"\quad f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}," \
        r" \quad f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right \}" \
        r"\Longrightarrow \left \{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \left\{unique\right\}\right \}"

    # A linear diagram.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    d = Diagram([f, g])
    grid = DiagramGrid(d)

    assert libreOffice(grid) == "\\begin{array}{cc}\n" \
        "A & B \\\\\n" \
        " & C \n" \
        "\\end{array}\n"


def test_Modules():
    from sympy.polys.domains import QQ
    from sympy.polys.agca import homomorphism

    R = QQ.old_poly_ring(x, y)
    F = R.free_module(2)
    M = F.submodule([x, y], [1, x**2])

    assert libreOffice(F) == r"{\mathbb{Q}\left[x, y\right]}^{2}"
    assert libreOffice(M) == \
        r"\left< {\left[ {x},{y} \right]},{\left[ {1},{x^{2}} \right]} \right>"

    I = R.ideal(x**2, y)
    assert libreOffice(I) == r"\left< {x^{2}},{y} \right>"

    Q = F / M
    assert libreOffice(Q) == r"\frac{{\mathbb{Q}\left[x, y\right]}^{2}}{\left< {\left[ {x},{y} \right]},{\left[ {1},{x^{2}} \right]} \right>}"
    assert libreOffice(Q.submodule([1, x ** 3 / 2], [2, y])) == \
        r"\left< {{\left[ {1},{\frac{x^{3}}{2}} \right]} + {\left< {\left[ {x},{y} \right]},{\left[ {1},{x^{2}} \right]} \right>}},{{\left[ {2},{y} \right]} + {\left< {\left[ {x},{y} \right]},{\left[ {1},{x^{2}} \right]} \right>}} \right>"

    h = homomorphism(QQ.old_poly_ring(x).free_module(2), QQ.old_poly_ring(x).free_module(2), [0, 0])

    assert libreOffice(h) == r"{\left[\begin{matrix}0 & 0\\0 & 0\end{matrix}\right]} : {{\mathbb{Q}\left[x\right]}^{2}} \to {{\mathbb{Q}\left[x\right]}^{2}}"


def test_QuotientRing():
    from sympy.polys.domains import QQ
    R = QQ.old_poly_ring(x)/[x**2 + 1]

    assert libreOffice(
        R) == r"\frac{\mathbb{Q}\left[x\right]}{\left< {x^{2} + 1} \right>}"
    assert libreOffice(R.one) == r"{1} + {\left< {x^{2} + 1} \right>}"


def test_Tr():
    #TODO: Handle indices
    A, B = symbols('A B', commutative=False)
    t = Tr(A*B)
    assert libreOffice(t) == r'\mbox{Tr}\left(A B\right)'


def test_Adjoint():
    from sympy.matrices import MatrixSymbol, Adjoint, Inverse, Transpose
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert libreOffice(Adjoint(X)) == r'X^\dag'
    assert libreOffice(Adjoint(X + Y)) == r'\left(X + Y\right)^\dag'
    assert libreOffice(Adjoint(X) + Adjoint(Y)) == r'X^\dag + Y^\dag'
    assert libreOffice(Adjoint(X * Y)) == r'\left(X Y\right)^\dag'
    assert libreOffice(Adjoint(Y) * Adjoint(X)) == r'Y^\dag X^\dag'
    assert libreOffice(Adjoint(X ** 2)) == r'\left(X^{2}\right)^\dag'
    assert libreOffice(Adjoint(X) ** 2) == r'\left(X^\dag\right)^{2}'
    assert libreOffice(Adjoint(Inverse(X))) == r'\left(X^{-1}\right)^\dag'
    assert libreOffice(Inverse(Adjoint(X))) == r'\left(X^\dag\right)^{-1}'
    assert libreOffice(Adjoint(Transpose(X))) == r'\left(X^T\right)^\dag'
    assert libreOffice(Transpose(Adjoint(X))) == r'\left(X^\dag\right)^T'


def test_Hadamard():
    from sympy.matrices import MatrixSymbol, HadamardProduct
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert libreOffice(HadamardProduct(X, Y * Y)) == r'X \circ \left(Y Y\right)'
    assert libreOffice(HadamardProduct(X, Y) * Y) == r'\left(X \circ Y\right) Y'


def test_boolean_args_order():
    syms = symbols('a:f')

    expr = And(*syms)
    assert libreOffice(expr) == 'a \\wedge b \\wedge c \\wedge d \\wedge e \\wedge f'

    expr = Or(*syms)
    assert libreOffice(expr) == 'a \\vee b \\vee c \\vee d \\vee e \\vee f'

    expr = Equivalent(*syms)
    assert libreOffice(expr) == 'a \\equiv b \\equiv c \\equiv d \\equiv e \\equiv f'

    expr = Xor(*syms)
    assert libreOffice(expr) == 'a \\veebar b \\veebar c \\veebar d \\veebar e \\veebar f'



def test_imaginary():
    i = sqrt(-1)
    assert libreOffice(i) == r'i'


def test_builtins_without_args():
    assert libreOffice(sin) == r'\sin'
    assert libreOffice(cos) == r'\cos'
    assert libreOffice(tan) == r'\tan'
    assert libreOffice(log) == r'\log'
    assert libreOffice(Ei) == r'\operatorname{Ei}'
    assert libreOffice(zeta) == r'\zeta'

def test_latex_greek_functions():
    # bug because capital greeks that have roman equivalents should not use
    # \Alpha, \Beta, \Eta, etc.
    s = Function('Alpha')
    assert libreOffice(s) == r'A'
    assert libreOffice(s(x)) == r'A{\left (x \right )}'
    s = Function('Beta')
    assert libreOffice(s) == r'B'
    s = Function('Eta')
    assert libreOffice(s) == r'H'
    assert libreOffice(s(x)) == r'H{\left (x \right )}'

    # bug because sympy.core.numbers.Pi is special
    p = Function('Pi')
    # assert latex(p(x)) == r'\Pi{\left (x \right )}'
    assert libreOffice(p) == r'\Pi'

    # bug because not all greeks are included
    c = Function('chi')
    assert libreOffice(c(x)) == r'\chi{\left (x \right )}'
    assert libreOffice(c) == r'\chi'

def test_translate():
    s = 'alpha'
    assert translate(s) == "%"+s
    s = 'beta'
    assert translate(s) == "%"+s
    s = 'eta'
    assert translate(s) == "%"+s
    s = 'omicron'
    assert translate(s) == "%"+s
    s = 'pi'
    assert translate(s) == "%"+s
    s = 'LAMBDAHatDOT'
    assert translate(s) == r'dot hat %LAMBDA'

def test_other_symbols():
    from libreOffice import other_symbols
    for s in other_symbols:
        assert libreOffice(symbols(s)) == "%" + s

def test_modifiers():
    # Test each modifier individually in the simplest case (with funny capitalizations)
    assert libreOffice(symbols("xMathring")) == r"\mathring{x}"
    assert libreOffice(symbols("xCheck")) == r"\check{x}"
    assert libreOffice(symbols("xBreve")) == r"\breve{x}"
    assert libreOffice(symbols("xAcute")) == r"\acute{x}"
    assert libreOffice(symbols("xGrave")) == r"\grave{x}"
    assert libreOffice(symbols("xTilde")) == r"\tilde{x}"
    assert libreOffice(symbols("xPrime")) == r"{x}'"
    assert libreOffice(symbols("xddDDot")) == r"\ddddot{x}"
    assert libreOffice(symbols("xDdDot")) == r"\dddot{x}"
    assert libreOffice(symbols("xDDot")) == r"\ddot{x}"
    assert libreOffice(symbols("xBold")) == r"\boldsymbol{x}"
    assert libreOffice(symbols("xnOrM")) == r"\left\lVert{x}\right\rVert"
    assert libreOffice(symbols("xAVG")) == r"\left\langle{x}\right\rangle"
    assert libreOffice(symbols("xHat")) == r"\hat{x}"
    assert libreOffice(symbols("xDot")) == r"\dot{x}"
    assert libreOffice(symbols("xBar")) == r"\bar{x}"
    assert libreOffice(symbols("xVec")) == r"\vec{x}"
    assert libreOffice(symbols("xAbs")) == r"\left\lvert{x}\right\rvert"
    assert libreOffice(symbols("xMag")) == r"\left\lvert{x}\right\rvert"
    assert libreOffice(symbols("xPrM")) == r"{x}'"
    assert libreOffice(symbols("xBM")) == r"\boldsymbol{x}"
    # Test strings that are *only* the names of modifiers
    assert libreOffice(symbols("Mathring")) == r"Mathring"
    assert libreOffice(symbols("Check")) == r"Check"
    assert libreOffice(symbols("Breve")) == r"Breve"
    assert libreOffice(symbols("Acute")) == r"Acute"
    assert libreOffice(symbols("Grave")) == r"Grave"
    assert libreOffice(symbols("Tilde")) == r"Tilde"
    assert libreOffice(symbols("Prime")) == r"Prime"
    assert libreOffice(symbols("DDot")) == r"\dot{D}"
    assert libreOffice(symbols("Bold")) == r"Bold"
    assert libreOffice(symbols("NORm")) == r"NORm"
    assert libreOffice(symbols("AVG")) == r"AVG"
    assert libreOffice(symbols("Hat")) == r"Hat"
    assert libreOffice(symbols("Dot")) == r"Dot"
    assert libreOffice(symbols("Bar")) == r"Bar"
    assert libreOffice(symbols("Vec")) == r"Vec"
    assert libreOffice(symbols("Abs")) == r"Abs"
    assert libreOffice(symbols("Mag")) == r"Mag"
    assert libreOffice(symbols("PrM")) == r"PrM"
    assert libreOffice(symbols("BM")) == r"BM"
    assert libreOffice(symbols("hbar")) == r"\hbar"
    # Check a few combinations
    assert libreOffice(symbols("xvecdot")) == r"\dot{\vec{x}}"
    assert libreOffice(symbols("xDotVec")) == r"\vec{\dot{x}}"
    assert libreOffice(symbols("xHATNorm")) == r"\left\lVert{\hat{x}}\right\rVert"
    # Check a couple big, ugly combinations
    assert libreOffice(symbols('xMathringBm_yCheckPRM__zbreveAbs')) == r"\boldsymbol{\mathring{x}}^{\left\lvert{\breve{z}}\right\rvert}_{{\check{y}}'}"
    assert libreOffice(symbols('alphadothat_nVECDOT__tTildePrime')) == r"\hat{\dot{\alpha}}^{{\tilde{t}}'}_{\dot{\vec{n}}}"

def test_greek_symbols():
    from libreOffice import convert_symbols
    for i in convert_symbols:
        assert libreOffice(Symbol(i)) == r'%'+i

@XFAIL
def test_builtin_without_args_mismatched_names():
    assert libreOffice(CosineTransform) == r'\mathcal{COS}'

def test_builtin_no_args():
    assert libreOffice(Chi) == r'\operatorname{Chi}'
    assert libreOffice(gamma) == r'\Gamma'
    assert libreOffice(KroneckerDelta) == r'\delta'
    assert libreOffice(DiracDelta) == r'\delta'
    assert libreOffice(lowergamma) == r'\gamma'

def test_issue_6853():
    p = Function('Pi')
    assert libreOffice(p(x)) == r"\Pi{\left (x \right )}"

def test_Mul():
    e = Mul(-2, x + 1, evaluate=False)
    assert libreOffice(e) == r'- 2 \left(x + 1\right)'
    e = Mul(2, x + 1, evaluate=False)
    assert libreOffice(e) == r'2 \left(x + 1\right)'
    e = Mul(S.One/2, x + 1, evaluate=False)
    assert libreOffice(e) == r'\frac{1}{2} \left(x + 1\right)'
    e = Mul(y, x + 1, evaluate=False)
    assert libreOffice(e) == r'y \left(x + 1\right)'
    e = Mul(-y, x + 1, evaluate=False)
    assert libreOffice(e) == r'- y \left(x + 1\right)'
    e = Mul(-2, x + 1)
    assert libreOffice(e) == r'- 2 x - 2'
    e = Mul(2, x + 1)
    assert libreOffice(e) == r'2 x + 2'

def test_Pow():
    e = Pow(2, 2, evaluate=False)
    assert libreOffice(e) == r'2^{2}'


def test_issue_7180():
    assert libreOffice(Equivalent(x, y)) == r"x \equiv y"
    assert libreOffice(Not(Equivalent(x, y))) == r"x \not\equiv y"

if __name__ == "__main__":

    test_translate()
    test_greek_symbols()
    test_latex_basic()