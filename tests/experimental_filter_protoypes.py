__author__ = 'Davtoh'

from tesisfunctions import filterFactory,normsigmoid,graph_filter
from RRtoolbox.lib.arrayops import Bandpass
from sympy import symbols, diff,Eq, solve, dsolve, limit, oo
from sympy.solvers import solve_undetermined_coeffs
import sympy as sp
import numpy as np

class Bandpass_mod(Bandpass):
    name = "Bandpass*input"
    def __call__(self, values):
        return super(Bandpass_mod, self).__call__(values) * values

filters = []
alfa=5
filters.append(filterFactory(alfa, 100))
filters.append(filterFactory(-1*alfa, 100))
filters.append(filterFactory(alfa, 100,150))
filters.append(filterFactory(-1*alfa, 100,150))
filters.append(filterFactory(-1*alfa, 150,100))
filters.append(filterFactory(alfa, 150,100))
"""
x = symbols("x")
e = filters[0](x)
l1,l2 = limit(e,x,-oo),limit(e,x,oo)
ed = diff(e)
s = solve(Eq(ed,0),x)"""

graph_filter(filters,single=False,cols=3,legend=False,annotate=True)#,levels=np.linspace(-40, 40,81))
filters = []
filters.append(Bandpass(alfa, 100, 150))
filters.append(Bandpass_mod(alfa, 100, 150))
graph_filter(filters,single=True,cols=3,legend=True,annotate=True)#,levels = np.linspace(0, 256,256))