from tesisfunctions import Controlstdout

if __name__ == "__main__":
    with Controlstdout(True):
        print("is this in output?")

    with Controlstdout(False):
        print("this has to appear")