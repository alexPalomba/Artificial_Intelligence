import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

ctrs = 3 * np.random.normal(0, 1, (2, 2))

X, y = make_blobs(n_samples=100, centers=ctrs, n_features=2, cluster_std=1.0, shuffle=False, random_state=0)

y[y==0] = -1

c0 = plt.scatter(X[y==-1,0], X[y==-1,1], s=20, color='r', marker='x')
c1 = plt.scatter(X[y==1,0], X[y==1,1], s=20, color='b', marker='o')

plt.legend((c0,c1), ('Class -1', 'Class +1'), loc='upper right', scatterpoints=1, fontsize=11)

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Two simple clusters of random data')

plt.savefig('hw3.plot.pdf', bbox_inches='tight')
plt.show()