from __future__ import division
from builtins import range
from random import random
from time import time, sleep
import pylab as plt

def simulateTime(stable = 0, rand = 100):
    """
    Simulates process.

    :param stable: stable time over
    :param rand: value of time to randomize
    :return: Dtime # time duration

    code does this but without waiting::

        t1 = time()
        sleep(Dtime) # simulates time duration
        t2 = time()
        return t2-t1 # the same as Dtime
    """
    return stable + random() * rand # process time

Nsamples = 200

# there are two equations, the simplified and full equation
TmeanN_full = 0
TmeanN_sim = 0
Mt_full = [0]
Mt_sim = [0]
Tprocesses = [0]

for N in range(1,Nsamples):
    if N < Nsamples/5:
        T_process = simulateTime()
    else:
        T_process = 0.5
    TmeanN_full = (T_process + (N-1)*TmeanN_full)/N
    TmeanN_sim = (T_process + TmeanN_sim)/2
    Tprocesses.append(T_process)
    Mt_full.append(TmeanN_full)
    Mt_sim.append(TmeanN_sim)

plt.plot(Tprocesses, color='yellow', label='process time', lw=4)
plt.plot(Mt_sim, color='red', label='simplified mean', lw=4)
plt.plot(Mt_full, color='blue', label='over time', lw=4)
plt.legend(loc='upper left')
plt.ylabel('mean times')
plt.savefig('WeakRef_mean_process_times_{}.png'.format(Nsamples))
plt.show()
