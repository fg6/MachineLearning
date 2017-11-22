
import numpy as np
from sortedcontainers import SortedList
from scipy.stats import multivariate_normal

class NaiveBayes:
    #def __init__(self):
    #    pass

    def fit(self, X, Y):
        self.X = X
        self.Y = set(Y)

        self.Classes = set(Y) 
        self.Prior = {}
        self.G = {}
        # smoothing
        epsilon=0.001*np.identity(28)
       
        for c in self.Classes:
            Xc = X[Y==c]
            Mean = np.mean(Xc, axis=0,dtype=np.float64)
            Sigma = np.var(Xc,axis=0,dtype=np.float64)+0.001 
           
            self.G[c] = (Mean, Sigma)
            self.Prior[c] = float(len(Xc))/len(Y)            

    def predict(self, X):
        
        results=[]
        max_posterior = -1
        max_class = None
        c_posterior = np.zeros((X.shape[0], len(self.G)))
        for c in self.Classes:
            mean, sigma = self.G[c]
            c_posterior[:,c] = multivariate_normal.logpdf(X, mean, sigma) + np.log(self.Prior[c]) # add cov !

        #print(len(c_posterior), np.argmax(c_posterior, axis=1))
     

        return np.argmax(c_posterior, axis=1)

             

    def score(self, X, Y):
        results = self.predict(X)
        #for i,v in enumerate(Y):
        #    print(i,v,results[i])
        score = np.mean(results == Y)
        return score

class Bayes:
 

    def fit(self, X, Y, e=0.001):

        self.X = X
        self.Y = set(Y)
        N,D = X.shape

        self.Classes = set(Y) 
        self.Prior = {}
        self.G = {}
        # smoothing
        epsilon=e*np.identity(28)
       
        for c in self.Classes:
            Xc = X [ Y==c ]
            Mean = np.mean(Xc, axis=0, dtype=np.float64)
            #Sigma = np.var(Xc, axis=0, dtype=np.float64) + e
            Cov = np.cov(Xc.T)+ np.eye(D)*e
           
            self.G[c] = (Mean, Cov)
            self.Prior[c] = float(len(Xc))/len(Y)            

    def predict(self, X):
        results=[]
        max_posterior = -1
        max_class = None
        c_posterior = np.zeros((X.shape[0], len(self.G)))
        for c in self.Classes:
            mean, cov = self.G[c]
            c_posterior[:,c] = multivariate_normal.logpdf(X, mean, cov) + np.log(self.Prior[c])

        return np.argmax(c_posterior, axis=1)

        
    
    def score(self, X, Y):
        results = self.predict(X)
        score = np.mean(results == Y)
        return score
