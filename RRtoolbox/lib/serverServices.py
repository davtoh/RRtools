# -*- coding: utf-8 -*-
import socket
import threading
from errno import ECONNREFUSED
from functools import partial
from multiprocessing import Pool
from time import time

import numpy as np

from RRtoolbox.lib.root import TimeOutException, TransferExeption
from config import FLAG_DEBUG, serializer

__author__ = 'Davtoh'

host ='localhost'
port = 50007
addr = (host,port)
NUM_CORES = 4


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
    p = Pool(NUM_CORES)
    ping_host = partial(ping, host)
    return filter(bool, p.map(ping_host, range(1, 65536)))

class Conection:
    """
    represent a connection to interchange objects between servers and clients.
    """

    def __init__(self, conn):
        self.conn = conn
        self.len = 0

    def sendLen(self,length, timeout = None):
        dest = self.conn
        ans = "False"
        t1 = time()
        while ans != "True": # get length of recipient for length
            if ans=="False":
                txt = "({},)".format(length)
                dest.send(txt)
                if FLAG_DEBUG: print "size",txt,"sent"
                ans = dest.recvfrom(5)[0] # get True or False
                if FLAG_DEBUG: print "received",ans
            if timeout is not None and time()-t1 > timeout:
                raise TimeOutException("Timeout sending length")

    def getLen(self, timeout = None):
        source = self.conn
        size = None
        t1 = time()
        while not size: # sent length of recipient for length
            try:
                if FLAG_DEBUG: print "waiting size..."
                size = eval(source.recvfrom(1024)[0])
                if FLAG_DEBUG: print "received size", size
            except Exception as e:
                print e
            if isinstance(size,tuple):
                source.send("True")
            else:
                source.send("False")
            if timeout is not None and time()-t1 > timeout:
                raise TimeOutException("Timeout of {} receiving length".format(timeout))
        self.len = size[0]
        return size[0]

    def recvall(self):
        buf = []
        l = self.len
        while l:
            newbuf = self.conn.recv(l)
            if not newbuf: break
            buf.append(newbuf)
            l = self.len = l - len(newbuf)
        return "".join(buf)

    def send(self, obj):
        pass

    def rcv(self):
        pass

def initServer(addr):
    """
    Inits a simple server from address.

    :param addr: (host, port)
    :return: socket
    """
    # server
    # Symbolic name meaning all available interfaces
    # Arbitrary non-privileged port
    s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(addr)# (host, port) host ='', port = 50007
    s.listen(1)
    return s # conn, addr = s.accept()

def initClient(addr, timeout = None):
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
            s.connect(addr)# (host, port) host ='localhost', port = 50007
            return s
        except socket.error as e:
            if timeout is None:
                raise e
            elif e.errno != ECONNREFUSED:
                raise e
            if time()-t1 > timeout:
                raise TimeOutException("Timeout connecting to server in {}".format(addr))

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

def generateServer(host = host, to = 63342):
    """
    generates a simple Server in available address.

    :param to: until port.
    :return: socket, address
    """
    s = None
    while True:
        port = int(np.random.rand()*to)
        addr = (host,port)
        try:
            s = initServer(addr)
            return s,addr
        except:
            try:
                s.close()
            except:
                pass

def sendPickle(obj,addr = addr, timeout = None, threaded = False):
    """
    Send potentially any data using sockets.

    :param obj: packable object.
    :param addr: socket or address.
    :param timeout: NotImplemented
    :return: True if sent successfully, else Throw error.
    """
    notToClose = isinstance(addr,socket.socket)
    if notToClose:
        s = addr
        if FLAG_DEBUG: print "address is a connection"
    else:
        if FLAG_DEBUG: print "initializing Server at {}".format(addr)
        s = initServer(addr)
    def helper():
        try:
            s.settimeout(timeout)
            if FLAG_DEBUG: print "waiting for connection..."
            conn, addr1 = s.accept()
            s.settimeout(None)
            if FLAG_DEBUG: print "connection accepted.."
            tosend = serializer.dumps(obj)
            if FLAG_DEBUG: print "waiting to confirm sending len"
            Conection(conn).sendLen(len(tosend),timeout=timeout)
            if FLAG_DEBUG: print "sending data"
            conn.send(tosend)
            return True
        except Exception as e:
            raise e
        finally:
            if not notToClose: # do not close if it was not opened in function.
                try:
                    s.close() # tries to close socket
                except:
                    pass
    if threaded:
        t = threading.Thread(target=helper)
        t.daemon = True
        t.start()
    else:
        return helper()

def rcvPickle(addr=addr, timeout = None):
    """
    Receive potentially any data using sockets.

    :param addr: socket or address.
    :param timeout: NotImplemented
    :return: data, else throws error.
    """
    notToClose = isinstance(addr,socket.socket)
    if notToClose:
        s = addr
    else:
        if FLAG_DEBUG: print "initializing client at {}".format(addr)
        s = initClient(addr, timeout)
    try:
        #s.settimeout(timeout)
        if FLAG_DEBUG: print "creating connection..."
        c = Conection(s)
        length = c.getLen(timeout=timeout)
        if FLAG_DEBUG: print "loading data..."
        rcvdata = c.recvall()
        if len(rcvdata) != length:
            raise TransferExeption("Data was transferred incomplete. Expected {} and got {} bytes".format(length, len(rcvdata)))
        if FLAG_DEBUG: print "received data of len {}".format(len(rcvdata))
        data = serializer.loads(rcvdata)
        s.close()
        return data
    except Exception as e:
        raise e
    finally:
        if not notToClose: # do not close if it was not opened in function.
            try:
                s.close() # tries to close socket
            except:
                pass

def string_is_socket_address(string):
    try:
        host,addr = string.split(":")
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
    if isinstance(string,basestring):
        host,addr = string.split(":")
        return rcvPickle((host,int(addr)),timeout=timeout)
    else:
        return [parseString(i, timeout=timeout) for i in string]

if __name__ == "__main__":
    initClient(addr,3)
    #print generateServer(to = 63342)