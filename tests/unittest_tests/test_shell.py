from RRtoolbox.shell import shell
import unittest

def fun1(a,b,c):
    pass

def fun2(a,b,c=3):
    pass

def fun3(a=1,b=2,c=3):
    pass

def fun4(a, b = 2, *args, **kwargs):
    pass

s = shell()
d1 = s.generateParser(fun1)
#d1.parse_args("1 2 3".split())
d2 = s.generateParser(fun2)
#d2.parse_args("1 2".split())
#d2.parse_args("1 2 --c 3".split())
d3 = s.generateParser(fun3)
#d3.parse_args("--a 1 --b 2 --c 3".split())
d4 = s.generateParser(fun4)
#d4.parse_args("".split())