# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from builtins import filter
from builtins import range
from past.builtins import basestring
from builtins import object
import socket
import threading
from errno import ECONNREFUSED
from functools import partial
from multiprocessing import Pool
from time import time

import numpy as np

from .root import TimeOutException, TransferExeption, NO_CPUs
from .config import FLAG_DEBUG, serializer, PY3


# read more https://docs.python.org/2/howto/sockets.html
host = 'localhost'
port = 50007
addr = (host, port)


def ping(host, port):
    """
    Ping to.

    :param host: IP address
    :param port: port address
    :return:
    """
    try:
        socket.socket().connect((host, port))
        return port
    except socket.error as err:
        if err.errno == ECONNREFUSED:
            return False
        raise


def scan_ports(host):
    """
    Scan opened ports in address.

    :param host: host IP to filter opened ports.
    :return: generator
    """
    # http://codereview.stackexchange.com/questions/38452/python-port-scanner
    p = Pool(NO_CPUs)
    ping_host = partial(ping, host)
    return list(filter(bool, p.map(ping_host, list(range(1, 65536)))))


class Connection(object):
    """
    represent a connection to interchange objects between servers and clients.
    """

    def __init__(self, conn):
        self.conn = conn
        self.len = 0

    def sendLen(self, length, timeout=None):
        dest = self.conn
        ans = "False"
        t1 = time()
        while ans != "True":  # get length of recipient for length
            if ans == "False":
                txt = "({},)".format(length)
                if PY3:
                    dest.sendall(txt.encode("utf-8"))
                else:
                    dest.sendall(txt)
                if FLAG_DEBUG:
                    print("size", txt, "sent")
                if PY3:
                    ans = dest.recvfrom(5)[0].decode("utf-8")  # get True or False
                else:
                    ans = dest.recvfrom(5)[0]  # get True or False
                if FLAG_DEBUG:
                    print("received", ans)
            if timeout is not None and time() - t1 > timeout:
                raise TimeOutException("Timeout sending length")

    def getLen(self, timeout=None):
        source = self.conn
        size = None
        t1 = time()
        while not size:  # sent length of recipient for length
            try:
                if FLAG_DEBUG:
                    print("waiting size...")
                if PY3:
                    size = eval(source.recvfrom(1024)[0].decode("utf-8"))
                else:
                    size = eval(source.recvfrom(1024)[0])
                if FLAG_DEBUG:
                    print("received size", size)
            except Exception as e:
                print(e)
            if PY3:
                if isinstance(size, tuple):
                    source.send("True".encode("utf-8"))
                else:
                    source.send("False".encode("utf-8"))
            else:
                if isinstance(size, tuple):
                    source.send("True")
                else:
                    source.send("False")
            if timeout is not None and time() - t1 > timeout:
                raise TimeOutException(
                    "Timeout of {} receiving length".format(timeout))
        self.len = size[0]
        return size[0]

    def recvall(self):
        buf = []
        l = self.len
        while l:
            newbuf = self.conn.recv(l)
            if not newbuf:
                break
            buf.append(newbuf)
            l = self.len = l - len(newbuf)
        if PY3:
            return b"".join(buf)
        else:
            return "".join(buf)

    def send(self, obj):
        pass

    def rcv(self):
        pass


def init_server(addr):
    """
    Init a simple server from address.

    :param addr: (host, port)
    :return: socket
    """
    # server
    # Symbolic name meaning all available interfaces
    # Arbitrary non-privileged port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(addr)  # (host, port) host ='', port = 50007
    s.listen(1)
    return s  # conn, addr = s.accept()


def init_client(addr, timeout=None):
    """
    Inits a simple client from address.

    :param addr: (host, port)
    :return: socket
    """
    # client
    # The remote host
    # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    t1 = time()
    while True:
        try:
            s.connect(addr)  # (host, port) host ='localhost', port = 50007
            return s
        except socket.error as e:
            if timeout is None:
                raise e
            elif e.errno != ECONNREFUSED:
                raise e
            if time() - t1 > timeout:
                raise TimeOutException(
                    "Timeout connecting to server in {}".format(addr))


def send_from(viewable, socket):
    """
    Send from viewable object.

    :param viewable: viewable object
    :param socket: destine socket
    :return: None
    """
    view = memoryview(viewable)
    while len(view):
        nsent = socket.send(view)
        view = view[nsent:]


def recv_into(viewable, socket):
    """
    Receive from socket into viewable object.

    :param viewable: viewable object
    :param socket: source socket
    :return: None
    """
    view = memoryview(viewable)
    while len(view):
        nrecv = socket.recv_into(view)
        view = view[nrecv:]


def generateServer(host=host, max_port=63342, tries=3):
    """
    generates a simple Server in available address.

    :param max_port: randomize socket from 1000 to max_port.
    :param tries: number of tries, if None it is indefinitely
    :return: socket, address
    """
    assert max_port > 1000
    assert tries >= 0
    s = None
    i = -1
    while i < tries:
        if tries is not None:
            i+=1
        port = int(1000 + np.random.rand() * (max_port - 1000))
        addr = (host, port)
        try:
            s = init_server(addr)
            return s, addr
        except:
            try:
                s.close()
            except:
                pass


def sendPickle(obj, addr=addr, timeout=None, threaded=False):
    """
    Send potentially any data using sockets.

    :param obj: packable object.
    :param addr: socket or address.
    :param timeout: NotImplemented
    :return: True if sent successfully, else Throw error.
    """
    notToClose = isinstance(addr, socket.socket)
    if notToClose:
        s = addr
        if FLAG_DEBUG:
            print("address is a connection")
    else:
        if FLAG_DEBUG:
            print("initializing Server at {}".format(addr))
        s = init_server(addr)

    def helper():
        try:
            s.settimeout(timeout)
            if FLAG_DEBUG:
                print("waiting for connection...")
            conn, addr1 = s.accept()
            s.settimeout(None)
            if FLAG_DEBUG:
                print("connection accepted..")
            tosend = serializer.dumps(obj)
            if FLAG_DEBUG:
                print("waiting to confirm sending len")
            Connection(conn).sendLen(len(tosend), timeout=timeout)
            if FLAG_DEBUG:
                print("sending data")
            conn.sendall(tosend)
            return True
        except Exception as e:
            raise e
        finally:
            if not notToClose:  # do not close if it was not opened in function.
                try:
                    s.close()  # tries to close socket
                except:
                    pass
    if threaded:
        t = threading.Thread(target=helper)
        # deamon is False to let the process finish
        # even if the program reached the end making it wait.
        t.daemon = False
        t.start()
    else:
        return helper()


def rcvPickle(addr=addr, timeout=None):
    """
    Receive potentially any data using sockets.

    :param addr: socket or address.
    :param timeout: NotImplemented
    :return: data, else throws error.
    """
    notToClose = isinstance(addr, socket.socket)
    if notToClose:
        s = addr
    else:
        if FLAG_DEBUG:
            print("initializing client at {}".format(addr))
        s = init_client(addr, timeout)
    try:
        # s.settimeout(timeout)
        if FLAG_DEBUG:
            print("creating connection...")
        c = Connection(s)
        length = c.getLen(timeout=timeout)
        if FLAG_DEBUG:
            print("loading data...")
        rcvdata = c.recvall()
        if len(rcvdata) != length:
            raise TransferExeption(
                "Data was transferred incomplete. Expected {} and got {} bytes".format(length, len(rcvdata)))
        if FLAG_DEBUG:
            print("received data of len {}".format(len(rcvdata)))
        data = serializer.loads(rcvdata)
        s.close()
        return data
    except Exception as e:
        raise e
    finally:
        if not notToClose:  # do not close if it was not opened in function.
            try:
                s.close()  # tries to close socket
            except:
                pass


def string_is_socket_address(string):
    try:
        host, addr = string.split(":")
        int(addr)
        return True
    except:
        return False


def parseString(string, timeout=3):
    """

    :param string:
    :param timeout:
    :return:
    """
    if isinstance(string, basestring):
        host, addr = string.split(":")
        return rcvPickle((host, int(addr)), timeout=timeout)
    else:
        return [parseString(i, timeout=timeout) for i in string]


#if __name__ == "__main__":
    #init_client(addr, 3)
    # print generateServer(to = 63342)
