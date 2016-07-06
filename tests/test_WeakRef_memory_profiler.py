"""
Demonstrates the sys.getsizeof (to get memory size of an object) with numpy arrays in comparison with the actual
computer RAM memory used by the program. It is to confirm that sys.getsizeof as a memory profiler fro WeakREf.py
"""

import os, datetime
import pylab as plt
import sys
import psutil
import numpy as np
# http://nbviewer.jupyter.org/url/fa.bianp.net/blog/static/code/2013/memory_usage.ipynb
# http://fa.bianp.net/blog/2013/different-ways-to-get-memory-consumption-or-lessons-learned-from-memory_profiler/

def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)
    return mem

mem_sys = []
mem_psutil = []
r = np.arange(10)+1

# BENCH MARKS
mem_psutil1 = memory_usage_psutil()
start = datetime.datetime.now()
for i in r:
    a = np.ones((100*i, 100*i))
    mem_psutil.append(memory_usage_psutil()-mem_psutil1)
psutil_timing = (datetime.datetime.now() - start).total_seconds()

start = datetime.datetime.now()
for i in r:
    a = np.ones((100*i, 100*i))
    mem_sys.append(sys.getsizeof(a) / float(2 ** 20)) # getting MB
sys_timing = (datetime.datetime.now() - start).total_seconds()

barWidth = 5
ranges = np.arange(2)+1
plt.bar(ranges, [psutil_timing, sys_timing], align='center')
plt.xticks(ranges, ['psutil', 'sys'])
plt.ylabel('Time (seconds)')
plt.title('Time making {} memory measurements'.format(len(r)))
plt.savefig('WeakRef_sys_vs_psutil_benchmark_{}_samples.png'.format(len(r)))
plt.show()

#norm_resource = np.array(mem_sys, np.float32)/np.sum(mem_sys)
#norm_psutil = np.array(mem_psutil, np.float32)/np.sum(mem_psutil)
print "used memory (sys): ",np.sum(mem_sys)
print "used memory (psutil): ",np.sum(mem_psutil)
print "pid: ",os.getpid()
plt.plot(mem_sys, color='blue', label='sys', lw=4)
plt.plot(mem_psutil, color='red', label='psutil', lw=4)
plt.legend(loc='upper left')
plt.ylabel('Memory usage in MB')
plt.savefig('WeakRef_sys_vs_psutil_{}_samples.png'.format(len(r)))
plt.show()
