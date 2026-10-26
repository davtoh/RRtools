import unittest
from RRtoolbox.shell import Shell


def fun1(a, b, c):
    pass


def fun2(a, b, c=3):
    pass


def fun3(a=1, b=2, c=3):
    pass


def fun4(a, b=2, *args, **kwargs):
    pass


class TestShell(unittest.TestCase):

    def test_generate_parsers(self):
        s = Shell()
        d1 = s.generateParser(fun1)
        self.assertIsNotNone(d1)
        d2 = s.generateParser(fun2)
        self.assertIsNotNone(d2)
        d3 = s.generateParser(fun3)
        self.assertIsNotNone(d3)
        d4 = s.generateParser(fun4)
        self.assertIsNotNone(d4)


if __name__ == '__main__':
    unittest.main()