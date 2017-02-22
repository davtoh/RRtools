#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (C) 2015-2017 David Toro <davsamirtor@gmail.com>
"""

"""
# compatibility with python 2 and 3
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

#__all__ = []
__author__ = "David Samir Toro"
#__copyright__ = "Copyright 2017, The <name> Project"
#__credits__ = [""]
__license__ = "GPL"
#__version__ = "1.0.0"
__maintainer__ = "David Samir Toro"
__email__ = "davsamirtor@gmail.com"
#__status__ = "Pre-release"

# import build-in modules
import sys

# import third party modules
try:
    import StringIO
except: # python 3
    import io as StringIO
import unittest
from RRtoolbox.lib.root import StdoutLOG
# replace stdout with StringIO
original_stdout = sys.stdout

def reset_stdout():
    """
     sys.stdout is reset to the original file
    """
    sys.stdout = original_stdout  # return original stdout

class TestStdoutLOG(unittest.TestCase):

    def test_with_all_contexts(self):
        """
        Test all contexts untouched
        """

        # TEST
        sys.stdout = stdout = StringIO.StringIO()

        with StdoutLOG(StringIO.StringIO(), add_stdout=True) as A:
            A_in = "Inside A"
            print(A_in,end="")

            with StdoutLOG(StringIO.StringIO(), add_stdout=True) as B:
                B_in = "Inside B"
                print(B_in,end="")

                with StdoutLOG(StringIO.StringIO(), add_stdout=True) as C:
                    C_in = "Inside C"
                    print(C_in,end="")
                    C._file.seek(0)
                    C_str = C._file.read()
                B._file.seek(0)
                B_str = B._file.read()

            with StdoutLOG(StringIO.StringIO(), add_stdout=True) as D:
                D_in = "Inside D"
                print(D_in,end="")
                D._file.seek(0)
                D_str = D._file.read()

            A._file.seek(0)
            A_str = A._file.read()
            A.close()

        # check sys.stdout is left untouched (with first StringIO)
        self.assertEqual(stdout,sys.stdout)

        # check A context
        self.assertEqual("".join([A_in,B_in,C_in,D_in]),A_str)

        # check B context
        self.assertEqual("".join([B_in,C_in]),B_str)

        # check C context
        self.assertEqual(C_in,C_str)

        # check C context
        self.assertEqual(D_in,D_str)

        # Uncomment to see what happened
        #reset_stdout()

        print("*"*10)
        print("After A")
        print(A_str)
        print("*" * 10)
        print("After B")
        print(B_str)
        print("*" * 10)
        print("After C")
        print(C_str)
        print("*" * 10)
        print("After D")
        print(D_str)

        reset_stdout()

    def test_with_not_stdout(self):
        """
        Test when context has add_stdout=False
        """

        # TEST
        sys.stdout = stdout = StringIO.StringIO()

        with StdoutLOG(StringIO.StringIO(), add_stdout=True) as A:
            A_in = "Inside A"
            print(A_in,end="")

            with StdoutLOG(StringIO.StringIO(), add_stdout=False) as B:
                B_in = "Inside B"
                print(B_in,end="")

                with StdoutLOG(StringIO.StringIO(), add_stdout=True) as C:
                    C_in = "Inside C"
                    print(C_in,end="")
                    C._file.seek(0)
                    C_str = C._file.read()
                B._file.seek(0)
                B_str = B._file.read()

            with StdoutLOG(StringIO.StringIO(), add_stdout=True) as D:
                D_in = "Inside D"
                print(D_in,end="")
                D._file.seek(0)
                D_str = D._file.read()

            A._file.seek(0)
            A_str = A._file.read()
            A.close()

        # check sys.stdout is left untouched (with first StringIO)
        self.assertEqual(stdout,sys.stdout)

        # check A context
        self.assertEqual("".join([A_in,D_in]),A_str)

        # Here B context do not add stdout leaving out A_in without logging B_in,C_in

        # check B context
        self.assertEqual("".join([B_in,C_in]),B_str)

        # check C context
        self.assertEqual(C_in,C_str)

        # check C context
        self.assertEqual(D_in,D_str)

        # Uncomment to see what happened
        #reset_stdout()

        print("*"*10)
        print("After A")
        print(A_str)
        print("*" * 10)
        print("After B")
        print(B_str)
        print("*" * 10)
        print("After C")
        print(C_str)
        print("*" * 10)
        print("After D")
        print(D_str)

        reset_stdout()

    def test_with_closed_context(self):
        """
        Test when context is closed before other contexts
        """

        # TEST
        sys.stdout = stdout = StringIO.StringIO()

        with StdoutLOG(StringIO.StringIO(), add_stdout=True) as A:
            A_in = "Inside A"
            print(A_in,end="")
            A._file.seek(0)
            A_str = A._file.read()
            A.close()

            with StdoutLOG(StringIO.StringIO(), add_stdout=True) as B:
                B_in = "Inside B"
                print(B_in,end="")

                with StdoutLOG(StringIO.StringIO(), add_stdout=True) as C:
                    C_in = "Inside C"
                    print(C_in,end="")
                    C._file.seek(0)
                    C_str = C._file.read()
                B._file.seek(0)
                B_str = B._file.read()

            with StdoutLOG(StringIO.StringIO(), add_stdout=True) as D:
                D_in = "Inside D"
                print(D_in,end="")
                D._file.seek(0)
                D_str = D._file.read()

        # check sys.stdout is left untouched (with first StringIO)
        self.assertEqual(stdout,sys.stdout)

        # check A context
        self.assertEqual(A_in,A_str)

        # Here A context stdout is closed not receiving B_in and C_in

        # check B context
        self.assertEqual("".join([B_in,C_in]),B_str)

        # check C context
        self.assertEqual(C_in,C_str)

        # check C context
        self.assertEqual(D_in,D_str)

        # Uncomment to see what happened
        #reset_stdout()

        print("*"*10)
        print("After A")
        print(A_str)
        print("*" * 10)
        print("After B")
        print(B_str)
        print("*" * 10)
        print("After C")
        print(C_str)
        print("*" * 10)
        print("After D")
        print(D_str)

        reset_stdout()

    def test_disorder_static(self):
        """
        Test when context is closed before other contexts
        """

        # TEST 4 possibilities in disorder
        sys.stdout = stdout = StringIO.StringIO()

        A = StdoutLOG(StringIO.StringIO(), add_stdout=False)
        # Should print to stdout
        A_in = "Inside A"
        print(A_in,end="")

        B = StdoutLOG(StringIO.StringIO(), add_stdout=False)
        # Should Not print to stdout
        B_in = "Inside B"
        print(B_in,end="")

        A._file.seek(0)
        A_str = A._file.read()
        A.close()

        self.assertEqual(B, sys.stdout)

        C = StdoutLOG(StringIO.StringIO(), add_stdout=False)
        C_in = "Inside C"
        print(C_in,end="")

        self.assertEqual(C, sys.stdout)

        D = StdoutLOG(StringIO.StringIO(), add_stdout=False)
        D_in = "Inside D"
        print(D_in,end="")

        D._file.seek(0)
        D_str = D._file.read()
        D.close()

        C._file.seek(0)
        C_str = C._file.read()
        C.close()

        B._file.seek(0)
        B_str = B._file.read()
        B.close()

        # check sys.stdout is left untouched (with first StringIO)
        self.assertEqual(stdout,sys.stdout)

        # Uncomment to see what happened
        reset_stdout()

        print("*"*10)
        print("After A")
        print(A_str)
        print("*" * 10)
        print("After B")
        print(B_str)
        print("*" * 10)
        print("After C")
        print(C_str)
        print("*" * 10)
        print("After D")
        print(D_str)

        reset_stdout()

    def test_disorder(self):
        """
        Test when context is closed before other contexts
        """
        import random

        # TEST N possibilities
        N = 500 # recursion error is reached around 1000
        sys.stdout = stdout = StringIO.StringIO()
        logs = []
        strings = []
        strings_confirm = [None for _ in range(N)]
        for tries in range(N):
            logs.append(StdoutLOG(StringIO.StringIO(),
                                  add_stdout=random.choice([False,True])))
            mystring = "String at try {}".format(tries)
            print(mystring)
            strings.append(mystring)
            if random.choice([0,1]):
                try:
                    index = int(random.random()*N)
                    log = logs[index]
                    if not log.closed:
                        log._file.seek(0)
                        strings_confirm[index] = log._file.read()
                        log.close()
                        #original_stdout.write("Cleaning log {}\n".format(index))
                except IndexError:
                    pass

        for index,log in enumerate(logs):
            if not log.closed:
                log._file.seek(0)
                strings_confirm[index] = log._file.read()
                log.close()
                #original_stdout.write("Cleaning log {}\n".format(index))


        # check sys.stdout is left untouched (with first StringIO)
        self.assertEqual(stdout,sys.stdout)

        references_count = len([log for log in logs if log.file_list])
        self.assertLess(references_count,3)

        # Uncomment to see what happened
        reset_stdout()
        #print(strings)
        #print(strings_confirm)
        #print("Circular references: {}".format(references_count))


if __name__ == '__main__':

    # test StdoutLOG class from root
    suite = unittest.TestLoader().loadTestsFromTestCase(TestStdoutLOG)
    unittest.TextTestRunner(verbosity=2).run(suite)