import numpy as np

from sklearn.svm import SVC
from plotter import prettyPicture
np.random.seed(seed=0)
n = 20
mean = [3, 15]
cov = [[1, 0], [0, 10]]
X1 = np.random.multivariate_normal(mean, cov, n).T
Y1 = [0]*n

mean2 = [6, 20]
cov2 = [[1, 3], [-2, 1]]
X2 = np.random.multivariate_normal(mean2, cov2, n).T
Y2 = [1]*n


clf = SVC(C=1, kernel='rbf')

XX = np.column_stack((X1, X2))
YY = Y1+Y2
clf.fit(XX.T, YY)

##import matplotlib.pyplot as plt
##
##plt.scatter(X1[0], X1[1], c='red')
##plt.scatter(X2[0], X2[1], c='blue')
##plt.axis('equal')
##plt.show()

prettyPicture(clf, XX.T, YY)
