import numpy as np
import random
import os, subprocess
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
 
class Perceptron:  
    #def __init__(self, N):
        # Random linearly separated data
        #xA,yA,xB,yB = [random.uniform(-1, 1) for i in range(4)]
        #self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
        #self.X = self.generate_points(N)
 
    def generate_points(self, N):
        X = []
        for i in range(N):
            x1,x2 = [random.uniform(-1, 1) for i in range(2)]
            x = np.array([1,x1,x2])
            s = int(np.sign(self.V.T.dot(x)))
            X.append((x, s))
        return X
        
    def generate_blobs(self):
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
 
    def classification_error(self, vec, pts=None):
        # Error defined as fraction of misclassified points
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
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
        w = np.zeros(3)
        X, N = self.X, len(self.X)
        it = 0
        bestW = copy.deepcopy(w)
        # Iterate until all points are correctly classified
        while self.classification_error(w) != 0:
            it += 1
            # Pick random misclassified point
            x, s = self.choose_miscl_point(w)
            # Update weights
            w += s*x
            if(self.classification_error(w)<self.classification_error(bestW)):
                bestW = copy.deepcopy(w)
            print(it)
            if save:
                self.plot(vec=bestW)
                plt.title('N = %s, Iteration %s\n' \
                          % (str(N),str(it)))
                plt.savefig('p_N%s_it%s' % (str(N),str(it)), \
                            dpi=200, bbox_inches='tight')
        self.w = w
 
    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)
		
p = Perceptron()
#p.pla(save = True)
p.generate_blobs()