import socket
from time import sleep
import numpy as np
import cPickle as pickle
from errno import ECONNREFUSED, EADDRINUSE
# sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
# https://www.safaribooksonline.com/library/view/python-cookbook-3rd/9781449357337/ch11s13.html
host ='localhost'
port = 50007
addr = (host,port)

class Conection:
    def __init__(self, conn):
        self.conn = conn
        self.len = 0

    def sendLen(self,length):
        dest = self.conn
        ans = "False"
        while ans != "True": # get length of recipient for length
            if ans=="False":
                txt = "({},)".format(length)
                dest.send(txt)
                #print "size",txt,"sended"
                ans = dest.recvfrom(1024)[0]
                #print "recived",ans

    def getLen(self):
        source = self.conn
        size = None
        while not size: # sent length of recipient for length
            try:
                size = eval(source.recvfrom(1024)[0])
                #print "received size", size
            except Exception as e:
                print e
            if isinstance(size,tuple):
                source.send("True")
            else:
                source.send("False")
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


def sendLen(conn, length):
    conn.send("({},)".format(length))

def getLen(conn, buff = 1024):
    return eval(conn.recvfrom(buff)[0])[0]

def initServer(addr):
    # server
    # Symbolic name meaning all available interfaces
    # Arbitrary non-privileged port
    s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(addr)# (host, port) host ='', port = 50007
    s.listen(1)
    return s # conn, addr = s.accept()

def initClient(addr):
    # client
    # The remote host
    # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(addr)# (host, port) host ='localhost', port = 50007
    return s

def send_from(arr, dest):
    view = memoryview(arr)
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]

def recv_into(arr, source):
    view = memoryview(arr)
    while len(view):
        nrecv = source.recv_into(view)
        view = view[nrecv:]

def send(obj, addr):
    # server
    # Symbolic name meaning all available interfaces
    # Arbitrary non-privileged port
    s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(addr)# (host, port) host ='', port = 50007
        s.listen(1)
        conn, addr = s.accept()
        tosend = pickle.dumps(obj)
        conn.send(tosend, conn)
    finally:
        s.close()

def rcv(addr,buf=1024):
    # client
    # The remote host
    # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(addr)# (host, port) host ='localhost', port = 50007
        buf = b''
        count = 16
        while count:
            newbuf = s.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        s.close()
        return pickle.loads(buf)
    finally:
        s.close()

def sender(arr, addr):
    # server
    # Symbolic name meaning all available interfaces
    # Arbitrary non-privileged port
    s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(addr)# (host, port) host ='', port = 50007
        s.listen(1)
        conn, addr = s.accept()
        ans = "False"
        while ans != "True":
            sleep(0.2)
            if ans=="False":
                txt = "{}".format(arr.shape)
                conn.send(txt)
                print "shape",txt,"sended"
                ans = conn.recvfrom(1024)[0]
                print "recived",ans
        print "shape comfirmed"
        print "sending array"
        send_from(arr.flatten(), conn)
    finally:
        print "closing port...."
        s.close()

def receiver(addr):
    # client
    # The remote host
    # The same port as used by the server
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print "waing connection"
        s.connect(addr)# (host, port) host ='localhost', port = 50007
        shape = None
        while not shape:
            sleep(0.2)
            try:
                shape = eval(s.recvfrom(1024)[0])
                print "received shape", shape
            except Exception as e:
                print e
            if isinstance(shape,tuple):
                s.sendall("True")
            else:
                s.sendall("False")
        print "reciver is building array"
        arr = np.zeros(shape).flatten()
        recv_into(arr, s)
        s.close()
        return arr, shape
    finally:
        print "closing port...."
        s.close()

def test_array_send(arr= (1,2,3,4,5)):
    tosend = np.array(arr)
    print "to send:"
    print tosend
    sender(tosend,addr)

def test_array_rcv():
    arr,shape = receiver(addr)
    print "Raw received"
    print arr,shape
    constructed = arr.reshape(shape)
    print "received"
    print constructed


def test_image_send():
    import cv2
    img = cv2.imread("/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/tests/_good.jpg")
    img = cv2.resize(img,(100,100))
    sender(img,addr)
    cv2.imshow("sended",img)
    cv2.waitKey()

def test_image_rcv():
    import cv2
    img2,shape = receiver(addr)
    img = img2.reshape(shape)
    cv2.imshow("received",img)
    cv2.waitKey()

def test_image_send_pickle():
    import cv2
    img = cv2.imread("/mnt/4E443F99443F82AF/Dropbox/PYTHON/RRtools/tests/_good.jpg")
    img = cv2.resize(img,(100,100))
    sendPickle(img,addr)
    cv2.imshow("sended",img)
    cv2.waitKey()

def test_image_rcv_pickle():
    import cv2
    img = rcvPickle(addr)
    cv2.imshow("received",img)
    cv2.waitKey()

def test_len_send(length=1234567898765432112345678987654321123456789876543211234567898765432112345678987654321):
    s = initServer(addr)
    conn, addr1 = s.accept()
    c = Conection(conn)
    c.sendLen(length)
    print "len sent: ",length,"with len",len(str(length))

def test_len_rcv():
    s = initClient(addr)
    c = Conection(s)
    length = c.getLen()
    print "len received: ",length,"with len",len(str(length))

def test_len_send0(length=1234567898765432112345678987654321123456789876543211234567898765432112345678987654321):
    s = initServer(addr)
    conn, addr1 = s.accept()
    sendLen(conn,length)
    print "len sent: ",length,"with len",len(str(length))

def test_len_rcv0():
    s = initClient(addr)
    length = getLen(s)
    print "len received: ",length,"with len",len(str(length))

def generateServer(to = 63342):
    s = None
    while True:
        port = np.random.rand()*to
        addr = (host,port)
        try:
            s = initServer(addr)
            return s,addr
        except:
            try:
                s.close()
            except:
                pass


def sendPickle(obj,addr = addr):
    s = initServer(addr)
    try:
        conn, addr1 = s.accept()
        tosend = pickle.dumps(obj)
        Conection(conn).sendLen(len(tosend))
        conn.send(tosend)
    finally:
        s.close()

def rcvPickle(addr=addr):
    s = initClient(addr)
    try:
        length = Conection(s).getLen()
        data = pickle.loads(s.recv(length))
        s.close()
        return data
    except:
        s.close()

if __name__ == "__main__":
    #test_len_send0()
    #
    #obj = np.array([1,2,3,4])
    #print "to send: ",obj
    #sendPickle(obj)
    test_image_send_pickle()