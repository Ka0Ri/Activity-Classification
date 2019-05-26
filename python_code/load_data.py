#pylint: skip-file
import os
import pandas as pd
import numpy as np
import random
path = "D:/final project/data/"
attribute = ["RLAaccX","RLAaccY","RLAaccZ","label"]
mapping = {1:0,2:1,3:2,5:3,6:4,7:5,8:6,31:7,32:8,33:9}
def load(path, attribute):
    natt = len(attribute)
    nrow = 200
    ndata = len(os.listdir(path))
    data = np.zeros([ndata, natt, nrow])
    i = 0
    for filename in os.listdir(path):
        df = pd.read_csv(path + filename)
        datapoint = (df[attribute]).as_matrix().transpose()
        data[i,:,:] = datapoint
        i = i + 1
    np.random.shuffle(data)
    nsample = (90*ndata)//100
    train_data = data[0:nsample,:,:]
    test_data = data[nsample:ndata,:,:]
    return [train_data, test_data]

def split10(data):
    size = np.shape(data)
    feature = data[:,0:size[1]-1,:]
    label = np.zeros([size[0], 10])
    for i in range(0,size[0]):
        cat = data[i,size[1] - 1,0]
        label[i, mapping[int(cat)]] = 1
    
    return [feature, label]

def split33(data):
    size = np.shape(data)
    feature = data[:,0:size[1]-1,:]
    label = np.zeros([size[0], 33])
    for i in range(0,size[0]):
        cat = data[i,size[1] - 1,0]
        label[i, int(cat) - 1] = 1
    
    return [feature, label]
# _, test = load(path, attribute)
# _, label = split(test)
# print(label)