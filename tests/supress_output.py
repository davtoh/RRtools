from __future__ import print_function
from RRtoolbox.lib.root import Controlstdout, TimeCode

if __name__ == "__main__":
    with Controlstdout(True):
        print("This should not appear")

    with Controlstdout(False):
        print("this has to appear")

    with Controlstdout(True):
        with TimeCode("Does it appears?"):
            pass

    with Controlstdout(True,buffer=open("supress_output_test","w+")) as a:
        print("This should not appear now, but will be printed afterwards")

    print("After Controlstdout:")
    print(a.buffered)