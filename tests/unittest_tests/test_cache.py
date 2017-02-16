from __future__ import print_function
from builtins import range
from builtins import object
from RRtoolbox.lib.cache import MemoizedDict,mapper
from RRtoolbox.lib.root import TimeCode
import unittest
from time import time
persistIn = "mydict"


class TextOp(object):
    # this is a main class
    # used in TestMemoizedDisc
    def __init__(self,val):
        self.val = val

tomemoize_processed = []
def tomemoize(val):
    """
    This is the doc of tomemoize. This function is intended to test Memoizers
    from cache module. internally it appends input variable "val" to the
    list tomemoize_processed, confirm that it was "processed".
    :param val: any value
    :return: val
    """
    tomemoize_processed.append(val) # this confirms if tomemize processed val
    return val

class TestMemoizer(unittest.TestCase):
    def test_works(self):
        pass

class TestMemoizedDisc(unittest.TestCase):

    def test_session(self):
        """
        test persistence to file
        :return:
        """
        mydict = MemoizedDict(persistIn).clear()
        del mydict

        mydict = MemoizedDict(persistIn)

        mydict["TextOp"] = TextOp(1)
        del mydict

        mydict = MemoizedDict(persistIn)
        self.assertEqual(mydict["TextOp"].val,1)

    def test_key_times(self):
        """
        test how much time memoizeDict takes in saving keys
        :return:
        """
        mydict = MemoizedDict(persistIn).clear()
        del mydict
        mydict = MemoizedDict(persistIn)
        secs = 2
        try:
            for i in range(1000):
                t1 = time()
                data = (("12"*100)*100)*100
                mydict[i] = data
                mytime = time()-t1
                self.assertTrue(mytime<=secs,"At added data No {} , it takes {} seg which is more than {} seg".format(i,mytime,secs))
        finally:
            with TimeCode("cleaning up..."):
                mydict.clear() # clean up

    def test_cleanup(self):
        """
        test how much time memoizeDict takes in cleaning up keys
        :return:
        """
        mydict = MemoizedDict(persistIn).clear()
        del mydict
        mydict = MemoizedDict(persistIn)
        secs = 5
        nokeys = 1000
        with TimeCode("adding {} keys...".format(nokeys)):
            print("")
            for i in range(nokeys):
                mydict[i] = i
                print("\rkey {}/{}".format(i+1,nokeys), end=' ')
        try:
            t1 = time()
            mydict.clear()
            mytime = time()-t1
            print("cleaning time: {}".format(mytime))
            self.assertTrue(mytime<=secs,"It took {} to eliminate {} keys where the indended time was {}".format(mytime,nokeys,secs))
        finally:
            mydict.clear() # clean up

    def test_failed_session(self):
        mydict = MemoizedDict(persistIn).clear()
        del mydict

        mydict = MemoizedDict(persistIn)

        class TextOpFail(object):
            # unfortunately all classes that are memoized must be present
            # as main classes and not inside other objects
            def __init__(self,val):
                self.val = val

        from pickle import PicklingError
        with self.assertRaises(PicklingError): # if used pickle
            mydict["TextopFail"] = TextOpFail(1)

def runTests(tests = None, verbosity=2):
    def filter_helper(test):
        return issubclass(test,unittest.TestCase)
    suite = unittest.TestLoader().loadTestsFromModule()
    unittest.TextTestRunner(verbosity=verbosity).run(suite)

if __name__ == '__main__':
    #suite = unittest.TestLoader().loadTestsFromModule(globals())
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoizedDisc)
    unittest.TextTestRunner(verbosity=2).run(suite)