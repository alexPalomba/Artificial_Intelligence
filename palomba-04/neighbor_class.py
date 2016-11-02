#Modified in class by Dr. Rivas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
from numpy import genfromtxt
from sklearn.model_selection import KFold
import time

def genDataSet(N):
    x = np.random.normal(0, 1, N)
    ytrue = (np.cos(x) + 2) / (np.cos(x * 1.4) + 2)
    noise = np.random.normal(0, 0.2, N)
    y = ytrue + noise
    return x, y, ytrue

def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5,5))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        V = self.V
        a, b = -V[1]/V[2], -V[0]/V[2]
        l = np.linspace(-1,1)
        plt.plot(l, a*l+b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x,s in self.X:
            plt.plot(x[1], x[2], cols[s]+'o')
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s]+'.')
        if vec != None:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                          % (str(len(self.X)),str(len(mispts))))
            plt.savefig('p_N%s' % (str(len(self.X))), \
                        dpi=200, bbox_inches='tight')

# read digits data & split it into X (training input) and y (target output)
X, y, ytrue = genDataSet(1000)
#y = dataset[:, 0]
#X = dataset[:, 1:]
#y[y!=0] = -1    #rest of numbers are negative class
#y[y==0] = +1    #number zero is the positive class
bestk=[]
kc=0
for n_neighbors in range(1,900,2):
  kf = KFold(n_splits=10)
  #n_neighbors = 85
  kscore=[]
  k=0
  for train, test in kf.split(X):
    #print("%s %s" % (train, test))
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    X_train = X_train.reshape((len(X_train),1))
    X_test = X_test.reshape((len(X_test),1))
    y_train = y_train.reshape((len(y_train),1))
    y_test = y_test.reshape((len(y_test),1))
    
    #time.sleep(100)
    # we create an instance of Neighbors Classifier and fit the data.
    clf = neighbors.KNeighborsRegressor(n_neighbors, weights='distance')
    #print(X_train.shape)
    #print(y_train.shape)
    clf.fit(X_train, y_train)
    kscore.append(abs(clf.score(X_test,y_test)))
    #print kscore[k]
    k=k+1
  #print (n_neighbors)
  bestk.append(sum(kscore)/len(kscore))
  #print (bestk[kc])
  kc+=1
# to do here: given this array of E_outs in CV, find the max, its 
# corresponding index, and its corresponding value of n_neighbors
print(bestk)
X = X.reshape((len(X), 1))
idx = [0, 1, 2]
topk = bestk[0:len(idx)]
for i in range(3, len(bestk)):
    if topk[0] < bestk[i]:
        if topk[0] > topk[1]:
            if topk[1] > topk[2]:
                topk[2] = topk[1]
                idx[2] = idx[1]
            topk[1] = topk[0]
            idx[1] = idx[0]
        elif topk[0] > topk[2]:
            topk[2] = topk[0]
            idx[2] = idx[0]
        topk[0] = bestk[i]
        idx[0] = i
    elif topk[1] < bestk[i]:
        if topk[1] > topk[2]:
            topk[2] = topk[1]
            idx[2] = idx[1]
        topk[1] = bestk[i]
        idx[1] = i
    elif topk[2] < bestk[i]:
        topk[2] = bestk[i]
        idx[2] = i
for i in range(0, len(idx)):
    idx[i] = 2 * idx[i] + 1
print(topk)
print(idx)
if topk[0] < topk[1]:
    topk[0] = topk[1]
    idx[0] = idx[1]
if topk[0] < topk[2]:
    idx[0] = idx[2]
clf = neighbors.KNeighborsRegressor(idx[2], weights='distance')
clf.fit(X,y)

plt.plot(X,y,'.')
plt.plot(X,ytrue,'rx')
#plt.plot(X,yhat,'g+')
plt.show

print ("Eout (R^2)" + str(clf.score(X,y)))
print("Eout true (R^2): " + str(clf.score(X,ytrue)))
#yhat = clf.predict(X)
