import numpy as np
import random
import os, subprocess
import matplotlib.pyplot as plt
import copy
from sklearn.datasets.samples_generator import make_blobs

class Pocket:
    def __init__(self, N):
        # Random linearly separated data
        self.X = self.generate_blobs(N)
    def generate_blobs(self, N):
        np.random.seed(13)
        ctrs = 3 * np.random.normal(0, 1, (2, 2))
        X, y = make_blobs(n_samples=N, centers=ctrs, n_features=2, cluster_std=1.0, shuffle=False, random_state=0)
        y[y==0] = -1
        c0 = plt.scatter(X[y==-1,0], X[y==-1,1], s=20, color='r', marker='x')
        c1 = plt.scatter(X[y==1,0], X[y==1,1], s=20, color='b', marker='o')
        plt.legend((c0,c1), ('Class -1', 'Class +1'), loc='upper right', scatterpoints=1, fontsize=11)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(r'Two simple clusters of random data')
        plt.savefig('hw3.plot.pdf', bbox_inches='tight')
        plt.show()
        bX = []
        for k in range(0,N) :
            bX.append((np.concatenate(([1], X[k,:])), y[k]))        
        # this will calculate linear regression at this point
        X = np.concatenate((np.ones((N,1)), X),axis=1); # adds the 1 constant
        self.linRegW = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)   # lin reg
        print (self.linRegW)
        return bX
        
    def generate_points(self, N):
        X, y = self.make_semi_circles(n_samples=N, sep=1)
        bX = []
        for k in range(0,N) :
            bX.append((np.concatenate(([1], X[k,:])), y[k]))        
        # this will calculate linear regression at this point
        X = np.concatenate((np.ones((N,1)), X),axis=1); # adds the 1 constant
        self.linRegW = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)   # lin reg
        print (self.linRegW)
        return bX
        
    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5,5))
        #plt.xlim(-1.5,2.5)
        #plt.ylim(-2.0,1.5)
        l = np.linspace(-1.5,2.5)
        V = self.linRegW
        a, b = -V[1]/V[2], -V[0]/V[2]
        plt.plot(l, a*l+b, 'k-')
        V = self.bestW    # for Pocket
        a, b = -V[1]/V[2], -V[0]/V[2]
        plt.plot(l, a*l+b, 'r-')
        cols = {1: 'r', -1: 'b'}
        for x,s in self.X:
            plt.plot(x[1], x[2], cols[s]+'.')
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s]+'x')
        if vec.size:
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
 
    def classification_error(self, vec, pts=None):
        # Error defined as fraction of misclassified points
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        #myErr = 0
        for x,s in pts:
            #myErr += abs(s - int(np.sign(vec.T.dot(x))))
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        #print error
        #print myErr
        return error
 
    def choose_miscl_point(self, vec):
        # Choose a random point among the misclassified
        pts = self.X
        mispts = []
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0,len(mispts))]
 
    def pla(self, save=False):
        # Initialize the weigths to zeros
        #w = np.zeros(3)
        w = np.array([ 0.86745859, 0.44992245, -0.18256571])
        self.bestW = copy.deepcopy(w);     # for Pocket
        self.plaError = []
        self.pocketError = []   # for Pocket
        X, N = self.X, len(self.X)
        it = 0
        stopIT = 0
        # Iterate until all points are correctly classified
        self.plaError.append(self.classification_error(w))
        self.pocketError.append(self.plaError[it])    # for Pocket
        while self.plaError[it] != 0 and it-stopIT < 10000:
            it += 1
            # Pick random misclassified point
            x, s = self.choose_miscl_point(w)
            # Update weights
            w += s*x
            self.plaError.append(self.classification_error(w))
            if (self.pocketError[it-1] > self.plaError[it]):  # for Pocket
                self.pocketError.append(self.plaError[it])
                self.bestW = copy.deepcopy(w);
                stopIT = it
            else:
                self.pocketError.append(self.pocketError[it-1])
            if save:
                self.plot(vec=w)
                plt.title('N = %s, Iteration %s\n' \
                          % (str(N),str(it)))
                plt.savefig('p_N%s_it%s' % (str(N),str(it)), \
                            dpi=200, bbox_inches='tight')
                plt.close()
        self.w = w
        print (self.plaError)
        print (self.pocketError)
        print (stopIT)
        print (self.classification_error(self.linRegW))
        return it
 
    def check_error(self, M, vec):
        check_pts = self.generate_blobs(M)
        return self.classification_error(vec, pts=check_pts)

def main():
    it = np.zeros(1)
    for x in range(0, 1):
        p = Pocket(100)
        it[x] = p.pla(save=False)
        print (it)

    #n, bins, patches = plt.hist(it, 50, normed=1, facecolor='green', alpha=0.75)
    #plt.show()


    #p.plot()


main()