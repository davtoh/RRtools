# http://www.ams.org/samplings/feature-column/fcarc-svd
# http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.svd.html
import numpy as np
import pylab as plt

points = np.array([[-1.03, 0.74, -0.02, 0.51, -1.31, 0.99, 0.69, -0.12, -0.72, 1.11],
                   [-2.23, 1.61, -0.02, 0.88, -2.39, 2.02, 1.62, -0.35, -1.67, 2.46]])

U, s, V = np.linalg.svd(points, full_matrices=False)
print U.shape, V.shape, s.shape
s[1] = 0
points2 = np.dot(U, np.dot(np.diag(s), V))
points2.sort(1) # sort with respect to axis 1, that is to sort each column
plt.plot(points[0,:],points[1,:],'ro',points2[0,:],points2[1,:],'k')
plt.show()