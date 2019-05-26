#pylint: skip-file
import os
import pandas as pd
import numpy as np
import glob

path1 = "D:/final project/all_ideal/"
path2 = "D:/final project/all_self/"
datapath = "D:/final project/all_data/"
# act = [1, 2, 3, 5, 6, 7, 8, 31, 32, 33]
act = range(1,34)
index = 1
for path in [path1, path2]:
    for a in act:
        for filename in os.listdir(path + str(a)):
            print(path + str(a) + '/' + filename)
            df = pd.read_csv(path + str(a) + '/' + filename)
            df.to_csv(datapath + str(index) + '.csv', index=False)
            index = index + 1