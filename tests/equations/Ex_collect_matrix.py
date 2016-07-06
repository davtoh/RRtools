# http://stackoverflow.com/a/30238906/5288758
import sympy as sy
#Linear Interpolation function: V(x)
v_1, theta_1, v_2, theta_2, x, L = sy.symbols(
    "v_1, theta_1, v_2, theta_2, x, L")
a_1, a_2, a_3, a_4 = sy.symbols("a_1, a_2, a_3, a_4", real=True)
V = a_1*x**0 + a_2*x**1 + a_3*x**2 + a_4*x**3
#Solve for coefficients (a_1, a_2, a_3, a_4) with BC's: V(x) @ x=0, x=L
shape_coeffs = sy.solve([sy.Eq(v_1, V.subs({x:0})),
                      sy.Eq(theta_1, V.diff(x).subs({x:0})),
                      sy.Eq(v_2, V.subs({x:L})),
                      sy.Eq(theta_2, V.diff(x).subs({x:L}))],
                     (a_1, a_2, a_3, a_4))
V = V.subs(shape_coeffs)
V = sy.collect(sy.expand(V), (v_1, theta_1, v_2, theta_2))
N = sy.Matrix([V.coeff(v) for v in (v_1, theta_1, v_2, theta_2)]).transpose()
print(sy.latex(N))