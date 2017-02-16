from builtins import range
import time
from multiprocessing import Process, Pipe
# http://stackoverflow.com/a/24371607/5288758
import numpy as np
import matplotlib.pyplot as plt

class DataStreamProcess(Process):
    def __init__(self, connec, *args, **kwargs):
        self.connec = connec
        Process.__init__(self, *args, **kwargs)

    def run(self):
        random_gen = np.random.mtrand.RandomState(seed=127260)
        for _ in range(30):
            time.sleep(0.01)
            new_pt = random_gen.uniform(-1., 1., size=2)
            self.connec.send(new_pt)


def main():
    conn1, conn2  = Pipe()
    data_stream = DataStreamProcess(conn1)
    data_stream.start()

    plt.gca().set_xlim([-1, 1.])
    plt.gca().set_ylim([-1, 1.])
    plt.gca().set_title("Running...")
    plt.ion()

    pt = None
    while True:
        if not(conn2.poll(0.1)):
            if not(data_stream.is_alive()):
                break
            else:
                continue
        new_pt = conn2.recv()
        if pt is not None:
            plt.plot([pt[0], new_pt[0]], [pt[1], new_pt[1]], "bs:")
            plt.pause(0.001)
        pt = new_pt

    plt.gca().set_title("Terminated.")
    plt.draw()
    plt.show(block=True)

if __name__ == '__main__':
    main()