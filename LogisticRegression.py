import numpy as np
import math
import random

class LogisticRegression():
    def __init__(self, LR=0.01, K=10000):
        self.LR = LR
        self.K = K
        self.W = None
        self.prediction = None
        
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, train_features, train_target):
        num_classes = len(train_target.value_counts())
        num_columns = len(train_features.columns)
        W = np.zeros((num_classes, num_columns))

        # number of samples (M) -> rows(input)
        M = train_features.shape[0]

        # Bias (Y) -> numpy.zeros(M, num classes)
        Y = np.zeros((M, num_classes))

        # transfer 1-D array target to 2-D array
        for i,v in enumerate(train_target):
            Y[i,v] = 1
        
        costArray = []
        learning_rate = self.LR
        K = self.K
        
        # logistic regression algorithm
        for _ in range(K):
            logitScores = np.dot(train_features, np.transpose(W))
            p = self.__sigmoid(logitScores)
            cost = - 1/M * ((Y * np.log(p)) + ((1-Y) * np.log(1-p)))
            if not len(costArray) :
                costArray = cost.sum(axis=0)
            else:
                costArray = np.vstack((costArray, cost.sum(axis=0)))
            Delta = (learning_rate / M) * np.dot(np.transpose(p-Y), train_features)
            W = W - Delta
        
        self.W = W
        self.num_classes = num_classes
        
        return (K, costArray)
    
    def predict(self, features):
        
        if self.W is None:
            raise ValueError('No training data found')
        W = self.W
        prediction = np.zeros((features.shape[0], self.num_classes))
        features_array = np.array(features)
        
        prob = self.__sigmoid(np.dot(features_array, np.transpose(W)))
        for i, v in enumerate(prob):
            max_prob_index = np.argmax(v, axis=0)
            prediction[i, max_prob_index] = 1
        prediction = self.__flatten_prediction(prediction)
        self.prediction = prediction
        return prediction    
        
    def __flatten_prediction(self, prediction):
        flat = [None]*len(prediction)
        for i, v in enumerate(prediction):
            for j in range(len(v)):
                if(v[j] == 1):
                    flat[i] = j
                    break;
        return flat

