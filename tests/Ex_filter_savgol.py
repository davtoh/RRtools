import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
# http://stackoverflow.com/a/20642478/5288758
# http://scipy.github.io/devdocs/generated/scipy.signal.savgol_filter.html#scipy.signal.savgol_filter
# http://stackoverflow.com/a/28857249/5288758
x = np.linspace(0,2*np.pi,100)
y = np.sin(x) + np.random.random(100) * 0.2
yhat = savgol_filter(y, 51, 3) # window size 51, polynomial order 3

plt.plot(x,y)
plt.plot(x,yhat, color='red')
plt.show()