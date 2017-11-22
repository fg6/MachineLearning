
import numpy as np
from sortedcontainers import SortedList

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, train, truths):
        self.train = train
        self.truths = truths

    def wpredict(self, sample):
        max_dist=0
        
        results=np.zeros(len(sample))
        for i, img in enumerate(sample):  # test each image of sample       
            distances = SortedList(load=self.k)  
            
            for ii, x in enumerate(self.train):  # loop over all images of train
                distance = (img - x)
                distance = distance.dot(distance)

                if len(distances) < self.k:
                    distances.add( (distance, self.truths[ii] ) )              
                else:
                     if distance < distances[-1][0]:
                         del distances[-1]
                         distances.add( (distance, self.truths[ii] ) )              

            max_dist = distances[-1][0]
            votes={}
            for dist, truth in distances:
                votes[truth] = votes.get(truth, 0) + max_dist-dist 
               
            maxvote=sorted(votes.items(),key=lambda x: -x[1])[0][0]
            results[i] =  maxvote

        return results

    def dinamic_predict(self, sample):
        results=np.zeros(len(sample))
        for i, img in enumerate(sample):  # test each image of sample       
            distances = SortedList(load=self.k)  # can be 2D, sort on first element of each pair
            
            for ii, x in enumerate(self.train):  # loop over all images of train
                distance = (img - x)
                distance = distance.dot(distance)

                if len(distances) < self.k:
                    distances.add( (distance, self.truths[ii] ) )              
                else:
                     if distance < distances[-1][0]:
                         del distances[-1]
                         distances.add( (distance, self.truths[ii] ) )              

          
            votes={}
            for dist, truth in distances:
                votes[truth] = votes.get(truth, 0) + 1   # take votes[truth] or 0 (if votes[truth] not defined)
               
            maxvote=sorted(votes.items(),key=lambda x: -x[1])[0][0]
            results[i] =  maxvote

        return results
        
    def predict(self, sample):
        results=np.zeros(len(sample))
        for i, img in enumerate(sample):  # test each image of sample       
            distances={} 
            index=0
            for ii, x in enumerate(self.train):  # loop over all images of train
                distance = (img - x)
                distances[index] = (ii, distance.dot(distance))
                index+=1

            # now sort distances
            sorteddist=(sorted(distances.values(),key=lambda x:(x[1])))
            
            #take votes for closest k imgs
            votes={}
            for d in sorteddist[0:k]:
                if self.truths[d[0]] in votes.keys():
                    votes[self.truths[d[0]]] += 1
                else:
                    votes[self.truths[d[0]]] = 1
                    
            maxvote=sorted(votes.items(),key=lambda x: -x[1])[0][0]
            results[i] = maxvote

        return results

    def score(self, sample, truths):
        # this is slightly faster, but more memory needed: results = self.predict(sample)
        results = self.dinamic_predict(sample)
        #results = self.wpredict(sample)
        score = np.mean(results == truths)
        return score

    def wscore(self, sample, truths):
        results = self.wpredict(sample)
        score = np.mean(results == truths)
        return score

