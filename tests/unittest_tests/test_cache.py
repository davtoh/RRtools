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

class TestMemoizedDisc(unittest.TestCase):

    def setUp(self):
        """
        called before test
        """
        self.path = persistIn
        self.mydict = MemoizedDict(persistIn)

    def tearDown(self):
        """
        called after test
        """
        with TimeCode("cleaning up..."):
            self.mydict.clear()
            del self.mydict

    def test_session(self):
        """
        test persistence to file
        """
        mydict = self.mydict

        # original data
        to_memoize = TextOp(1)
        internal_data = to_memoize.val

        # memoize
        mydict["TextOp"] = to_memoize

        # simulate other session
        del mydict
        mydict = MemoizedDict(self.path)

        # test internal data is the same
        self.assertEqual(mydict["TextOp"].val, internal_data)

        # test objects are the same
        self.assertNotEqual(to_memoize, mydict["TextOp"])

    def test_key_times(self):
        """
        test how much time memoizeDict takes in saving keys when they are
        added each time.
        """
        mydict = self.mydict
        len_keys = 1000
        to_mean = 10
        data = (("12"*100)*100)*100

        # calculate expected time of writing
        time_write_expected = time()
        for i in range(1, to_mean):
            mydict[-i] = data
        time_write_expected = (time() - time_write_expected)/to_mean
        mydict.clear()

        for i in range(len_keys):
            time_write = time()
            mydict[i] = data
            time_write = time()-time_write

            # check in each iteration
            self.assertAlmostEqual(
                time_write,
                time_write_expected,
                delta=time_write_expected * 0.2, # permissive seconds
                msg="At added data No {}, it takes {} seg which is not close to "
                    "{} seg".format(i,time_write,time_write_expected)
            )

    def test_operation_times(self):
        """
        test how much time memoizeDict takes in writing and cleaning up keys
        :return:
        """
        mydict = self.mydict
        len_keys = 1000
        to_mean = 10

        # calculate expected time of writing
        time_write_expected = time()
        for i in range(1, to_mean):
            mydict[-i] = -i
        time_write_expected = (time() - time_write_expected)/to_mean

        # take writing time
        time_write = time()
        with TimeCode("adding {} keys...".format(len_keys)):
            print("")
            for i in range(len_keys):
                mydict[i] = i
                print("\rkey {}/{}".format(i+1,len_keys), end=' ')

        time_write = time() - time_write
        print("memoizing time: {}".format(time_write ))
        self.assertAlmostEqual(
            time_write,
            time_write_expected,
            delta=len_keys * 0.2, # permissive seconds
            msg="It took {} to create {} keys where the intended time was {}"
                .format(time_write, len_keys, time_write_expected)
        )

        # calculate expected time of eliminating keys
        time_cleanup_expected = time()
        for i in range(1, to_mean):
            del mydict[-i]
        time_cleanup_expected = ((time() - time_cleanup_expected) * len_keys)/to_mean

        # take eliminating time
        time_cleanup = time()
        mydict.clear()
        time_cleanup = time() - time_cleanup
        print("cleaning time: {}".format(time_cleanup))
        self.assertAlmostEqual(
            time_cleanup,
            time_cleanup_expected,
            delta=len_keys*0.2, # permissive seconds
            msg="It took {} to eliminate {} keys where the intended time was {}"
                .format(time_cleanup, len_keys, time_cleanup_expected)
        )

    def test_failed_session(self):
        """

        :return:
        """
        mydict = self.mydict

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
    unittest.main()
    #suite = unittest.TestLoader().loadTestsFromModule(globals())
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestMemoizedDisc)
    #unittest.TextTestRunner(verbosity=2).run(suite)