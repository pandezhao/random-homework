import numpy as np
import csv
import math

def getdata():
    csv_trainfile = csv.reader(open('train.csv','r'))
    result=[]
    for DNA in csv_trainfile:
        tmp = {'features': None ,'label': None}
        feature = []
        for i in DNA[1]:
            feature.append(onehot(i)) 
        tmp['features']=np.array(feature)
        tmp['label']=DNA[2]
        result.append(tmp)
    return result

def gettestdata():
    csv_trainfile = csv.reader(open('test.csv','r'))
    result=[]
    for DNA in csv_trainfile:
        tmp = {}
        feature = []
        for i in DNA[1]:
            feature.append(onehot(i)) 
        tmp['features']=np.array(feature)
        result.append(tmp)
    return result


def onehot(a):
    if a == 'A':
        return np.array([[1],[0],[0],[0]])
    if a == 'C':
        return np.array([[0],[1],[0],[0]])
    if a == 'G':
        return np.array([[0],[0],[1],[0]])
    if a == 'T':
        return np.array([[0],[0],[0],[1]])

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)            
    m = X.shape[0]                  
    mini_batches = []

    shuffled_X = X
    shuffled_Y = Y

    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k+1) * mini_batch_size,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches