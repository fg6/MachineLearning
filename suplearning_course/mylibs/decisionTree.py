
import numpy as np
from sortedcontainers import SortedList
from scipy.stats import multivariate_normal
from collections import Counter

def entropy(Y):
        N=len(Y)
        ones = (Y==1).sum()
        
        if ones == 0 or ones == N: # no entropy
            return 0
        
        p1 = float(ones) / N
        p0 = 1 - p1       
        return -p0*np.log2(p0) - p1*np.log2(p1)
    

class TreeNode:
    def __init__(self, max_depth=None, depth=0):
        self.maxdepth = max_depth
        self.depth = depth

        
    def myentropy(self, Y):
        N=len(Y)      
        if len(set(Y)) == 1:
            # no entropy
            return 0        
        entr=0
        for num in set(Y):  # or loop over all possible Y? 0-9?
            num_counts = (Y==num).sum()
            if num_counts !=0: 
                p = float(num_counts) / N
                entr -= p*np.log2(p)        
        return entr

  
    def entropy(self, Y):
        N=len(Y)
        ones = (Y==1).sum()
        
        if ones == 0 or ones == N: # no entropy
            return 0
        
        p1 = float(ones) / N
        p0 = 1 - p1       
        return -p0*np.log2(p0) - p1*np.log2(p1)
    

    def infgain(self, X, Y, split):
        yleft = Y[ X < split ]
        yright = Y[ X >= split ]

        if len(yleft) == 0 or len(yleft) == len(Y):
            # no information on this split
            return 0
        
        pleft = float(len(yleft))/len(Y)
        pright = 1 - pleft
        
        #ig =  entropy(Y) - pleft*entropy(yleft) - pright*entropy(yright)
        ig =  self.entropy(Y) - pleft*self.entropy(yleft) - pright*self.entropy(yright)
        #ig =  self.myentropy(Y) - pleft*self.myentropy(yleft) - pright*self.myentropy(yright)
        return ig
        
    def find_split(self, X, Y, c):
        thisX=X[:,c]
        sindex=np.argsort(thisX)
        thisX = thisX[sindex]
        thisY = Y[sindex]

        b_points = [ ii for ii in range(len(thisY)-1) if thisY[ii] != thisY[ii+1] ]
        
        best_ig=0
        best_split=0
        for split in b_points:
            split_point=(float(thisX[split]+thisX[split+1])/2)
            this_ig=self.infgain(thisX, thisY, split_point)
            if this_ig > best_ig:
                best_ig=this_ig
                best_split=split_point
       
        return best_ig, best_split
            

    def fit(self, X, Y):
        
        if len(Y) == 1 or len(set(Y)) == 1:
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = set(Y)
            
        else:
            cols=X.shape[1]  #number of dimenions
            best_col = None
            best_ig = 0
            best_split = None
            for col in range(cols):
                ig, split = self.find_split(X, Y, col)
                if ig > best_ig:
                    best_ig = ig
                    best_col = col
                    best_split = split
                    
            if best_ig == 0:  # no more split
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = Counter(Y.flat).most_common(1)[0][0]
            else:
                self.col = best_col
                self.split = best_split

                if self.maxdepth < self.depth:
                    Xleft = X[ X[:,self.col] < self.split ]
                    Xright = X[ X[:,self.col] >= self.split ]
                    Yleft = Y[ X[:,self.col] < self.split ]
                    Yright = Y[ X[:,self.col] >= self.split ]

                    self.left = TreeNode(self.maxdepth, self.depth+1)
                    self.left.fit(Xleft, Yleft)
                    self.right = TreeNode(self.maxdepth, self.depth+1)
                    self.right.fit(Xright, Yright)
                    
                else:                 
                    self.left = None
                    self.right = None
                    Yleft = Y[ X[:,self.col] < self.split ]
                    Yright = Y[ X[:,self.col] >= self.split ]
                    self.prediction = [Counter(Yleft.flat).most_common(1)[0][0],  Counter(Yright.flat).most_common(1)[0][0] ]
        print(self.col, self.split)
            
           
    def predict_one(self, X):
        if self.col is not None and self.split is not None:
            if X[self.col] < self.split: # left
                if self.left is not None:
                    prediction = self.left.predict_one(X)
                    #print " case 1, new calc:", self.left.col, self.left.split
                else:  # leaf
                    prediction = self.prediction[0]
                    #print " case 1", prediction,self.col, self.split
            else:
                if self.right is not None:
                    prediction = self.right.predict_one(X)
                    #print " case 2, new calc:", self.right.col, self.right.split
                else:  # leaf
                    prediction = self.prediction[1]  
                    #print " case 2", prediction,self.col, self.split
        else:
            prediction = self.prediction
            #print " case 3", prediction,self.col, self.split
     
        return prediction
 
            
    def predict(self, X):       
        results=[]
        ii=0
        for x in X:
            #print ii
            results.append(self.predict_one(x))
            ii+=1
        return results


    def score(self, X, Y):
        results = self.predict(X)
        #for i,v in enumerate(Y):
        #    print(i,v,results[i])
        score = np.mean(results == Y)
        return score

# This class is kind of redundant
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        #self.root = TreeNodeOrig(max_depth=self.max_depth)
        self.root = TreeNodeNew(max_depth=self.max_depth)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)



    
class TreeNodeNew:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X, Y):
        
        if len(Y) == 1 or len(set(Y)) == 1:
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]
            
        else:
            cols=X.shape[1]  #number of dimenions
            best_col = None
            best_ig = 0
            best_split = None
            for col in range(cols):
                ig, split = self.find_split(X, Y, col)
                if ig > best_ig:
                    best_ig = ig
                    best_col = col
                    best_split = split
                    
            if best_ig == 0:  # no more split
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = Counter(Y.flat).most_common(1)[0][0]
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None
                    Yleft = Y[ X[:,self.col] < self.split ]
                    Yright = Y[ X[:,self.col] >= self.split ]
                    self.prediction = [Counter(Yleft.flat).most_common(1)[0][0],  Counter(Yright.flat).most_common(1)[0][0] ]
                #if self.depth < self.max_depth:
                else:
                    Xleft = X[ X[:,self.col] < self.split ]
                    Xright = X[ X[:,self.col] >= self.split ]
                    Yleft = Y[ X[:,self.col] < self.split ]
                    Yright = Y[ X[:,self.col] >= self.split ]

                    self.left = TreeNodeNew(self.max_depth, self.depth+1)
                    self.left.fit(Xleft, Yleft)
                    self.right = TreeNodeNew(self.max_depth, self.depth+1)
                    self.right.fit(Xright, Yright)
                    
                   

        
    def fitTest(self, X, Y):  #testing
        
        if len(Y) == 1 or len(set(Y)) == 1:
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]          
        else:
            cols=X.shape[1]  #number of dimenions
            best_col = None
            best_ig = 0
            best_split = None
            for col in range(cols):
                ig, split = self.find_split(X, Y, col)
                if ig > best_ig:
                    best_ig = ig
                    best_col = col
                    best_split = split

           
            if best_ig == 0:  # no more split
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = Counter(Y.flat).most_common(1)[0][0]
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None
                    Yleft = Y[ X[:,self.col] < self.split ]
                    Yright = Y[ X[:,self.col] >= self.split ]
                    self.prediction = [Counter(Yleft.flat).most_common(1)[0][0],  Counter(Yright.flat).most_common(1)[0][0] ]
                else:                 
                    Xleft = X[ X[:,self.col] < self.split ]
                    Xright = X[ X[:,self.col] >= self.split ]
                    Yleft = Y[ X[:,self.col] < self.split ]
                    Yright = Y[ X[:,self.col] >= self.split ]

                    self.left = TreeNodeNew(self.max_depth, self.depth+1)
                    self.left.fit(Xleft, Yleft)
                    self.right = TreeNodeNew(self.max_depth, self.depth+1)
                    self.right.fit(Xright, Yright)

                   


    def fitOLD(self, X, Y):
        if len(Y) == 1 or len(set(Y)) == 1:
            # base case, only 1 sample
            # another base case
            # this node only receives examples from 1 class
            # we can't make a split
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]

        else:
            D = X.shape[1]
            cols = range(D)

            max_ig = 0
            best_col = None
            best_split = None
            for col in cols:
                ig, split = self.find_split(X, Y, col)
                # print "ig:", ig
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split

            if max_ig == 0:
                # nothing we can do
                # no further splits
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(Y.mean())
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(Y[X[:,best_col] < self.split].mean()),
                        np.round(Y[X[:,best_col] >= self.split].mean()),
                    ]
                else:
                    # print "best split:", best_split
                    left_idx = (X[:,best_col] < best_split)
                    # print "left_idx.shape:", left_idx.shape, "len(X):", len(X)
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNodeNew(self.depth + 1, self.max_depth)
                    self.left.fit(Xleft, Yleft)

                    right_idx = (X[:,best_col] >= best_split)
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = TreeNodeNew(self.depth + 1, self.max_depth)
                    self.right.fit(Xright, Yright)


                    

    def find_split(self, X, Y, c): #new
        thisX=X[:,c]
        sindex=np.argsort(thisX)
        thisX = thisX[sindex]
        thisY = Y[sindex]

        b_points = [ ii for ii in range(len(thisY)-1) if thisY[ii] != thisY[ii+1] ]
        
        best_ig=0
        best_split=0
        for split in b_points:
            split_point=(float(thisX[split]+thisX[split+1])/2)
            this_ig=self.information_gain(thisX, thisY, split_point)
            if this_ig > best_ig:
                best_ig=this_ig
                best_split=split_point
       
        return best_ig, best_split


    def information_gain(self, X, Y, split): #new
        yleft = Y[ X < split ]
        yright = Y[ X >= split ]

        if len(yleft) == 0 or len(yleft) == len(Y):
            # no information on this split
            return 0
        
        pleft = float(len(yleft))/len(Y)
        pright = 1 - pleft
        
        ig =  entropy(Y) - pleft*entropy(yleft) - pright*entropy(yright)
        return ig


    def predict_one(self, X): #new
        if self.col is not None and self.split is not None:
            if X[self.col] < self.split: # left
                if self.left is not None:
                    prediction = self.left.predict_one(X)
                    #print " case 1, new calc:", self.left.col, self.left.split
                else:  # leaf
                    prediction = self.prediction[0]
                    #print " case 1", prediction,self.col, self.split
            else:
                if self.right is not None:
                    prediction = self.right.predict_one(X)
                    #print " case 2, new calc:", self.right.col, self.right.split
                else:  # leaf
                    prediction = self.prediction[1]  
                    #print " case 2", prediction,self.col, self.split
        else:
            prediction = self.prediction
            #print " case 3", prediction,self.col, self.split
     
        return prediction
    

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        ii=0

        for i in xrange(N):
            #print ii
            P[i] = self.predict_one(X[i])
            ii+=1
        return P


    
    
class TreeNodeOrig:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X, Y):
        if len(Y) == 1 or len(set(Y)) == 1:
            # base case, only 1 sample
            # another base case
            # this node only receives examples from 1 class
            # we can't make a split
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]

        else:
            D = X.shape[1]
            cols = range(D)

            max_ig = 0
            best_col = None
            best_split = None
            for col in cols:
                ig, split = self.find_split(X, Y, col)
                # print "ig:", ig
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split

            if max_ig == 0:
                # nothing we can do
                # no further splits
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(Y.mean())
            else:
                self.col = best_col
                self.split = best_split

                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        np.round(Y[X[:,best_col] < self.split].mean()),
                        np.round(Y[X[:,best_col] >= self.split].mean()),
                    ]
                else:
                    # print "best split:", best_split
                    left_idx = (X[:,best_col] < best_split)
                    # print "left_idx.shape:", left_idx.shape, "len(X):", len(X)
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNodeOrig(self.depth + 1, self.max_depth)
                    self.left.fit(Xleft, Yleft)

                    right_idx = (X[:,best_col] >= best_split)
                    Xright = X[right_idx]
                    Yright = Y[right_idx]
                    self.right = TreeNodeOrig(self.depth + 1, self.max_depth)
                    self.right.fit(Xright, Yright)

    def find_split(self, X, Y, col):
        # print "finding split for col:", col
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]

        # Note: optimal split is the midpoint between 2 points
        # Note: optimal split is only on the boundaries between 2 classes

        # if boundaries[i] is true
        # then y_values[i] != y_values[i+1]
        # nonzero() gives us indices where arg is true
        # but for some reason it returns a tuple of size 1
        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0]
        best_split = None
        max_ig = 0
        for b in boundaries:
            split = (x_values[b] + x_values[b+1]) / 2
            ig = self.information_gain(x_values, y_values, split)
            if ig > max_ig:
                max_ig = ig
                best_split = split
        return max_ig, best_split

    def information_gain(self, x, y, split):
        # assume classes are 0 and 1
        # print "split:", split
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)
        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0 #float(len(y1)) / N
        # print "entropy(y):", entropy(y)
        # print "p0:", p0
        # print "entropy(y0):", entropy(y0)
        # print "p1:", p1
        # print "entropy(y1):", entropy(y1)
        return entropy(y) - p0*entropy(y0) - p1*entropy(y1)

    def predict_one(self, x):
        # use "is not None" because 0 means False
        if self.col is not None and self.split is not None:
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                    #print " case 1, new calc:", self.left.col, self.left.split
                else:
                    p = self.prediction[0]
                    #print " case 1", p,self.col, self.split
            else:
                if self.right:
                    p = self.right.predict_one(x)
                    #print " case 2, new calc:", self.right.col, self.right.split
                else:
                    p = self.prediction[1]
                    #print " case 2", p,self.col, self.split
        else:
            # corresponds to having only 1 prediction
            p = self.prediction
            #print " case 3", p,self.col, self.split
        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)
        ii=0

        for i in xrange(N):
            #print ii
            P[i] = self.predict_one(X[i])
            ii+=1
        return P
