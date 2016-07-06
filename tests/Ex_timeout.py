import signal
import time
# https://pythonadventures.wordpress.com/2012/12/08/raise-a-timeout-exception-after-x-seconds/
# http://code.activestate.com/recipes/534115-function-timeout/
# http://eli.thegreenplace.net/2011/08/22/how-not-to-set-a-timeout-on-a-computation-in-python
# http://eventlet.net/doc/modules/timeout.html
def test_request(arg=None):
    """Your http request."""
    time.sleep(2)
    return arg
 
class Timeout():
    """Timeout class using ALARM signal."""
    class Timeout(Exception):
        pass
 
    def __init__(self, sec):
        self.sec = sec
 
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout) # it seems it is UNIX only
        signal.alarm(self.sec)
 
    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm
 
    def raise_timeout(self, *args):
        raise Timeout.Timeout()
 
def main():
    # Run block of code with timeouts
    try:
        with Timeout(3):
            print test_request("Request 1")
        with Timeout(1):
            print test_request("Request 2")
    except Timeout.Timeout:
        print "Timeout"

if __name__ == "__main__":
    main()