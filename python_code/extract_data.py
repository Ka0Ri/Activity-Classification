#pylint: skip-file
import os
import pandas as pd
import numpy as np
import glob

header = ["time(second)","time(milisecond)",
            "RLAaccX","RLAaccY","RLAaccZ","RLAgyrX","RLAgyrY","RLAgyrZ","RLAmagX","RLAmagY","RLAmagZ","RLAq1","RLAq2","RLAq3","RLAq4",
            "RUAaccX","RUAaccY","RUAaccZ","RUAgyrX","RUAgyrY","RUAgyrZ","RUAmagX","RUAmagY","RUAmagZ","RUAq1","RUAq2","RUAq3","RUAq4",
            "BACKaccX","BACKaccY","BACKaccZ","BACKgyrX","BACKgyrY","BACKgyrZ","BACKmagX","BACKmagY","BACKmagZ","BACKq1","BACKq2","BACKq3","BACKq4",
            "LUAaccX","LUAaccY","LUAaccZ","LUAgyrX","LUAgyrY","LUAgyrZ","LUAmagX","LUAmagY","LUAmagZ","LUAq1","LUAq2","LUAq3","LUAq4",
            "LLAaccX","LLAaccY","LLAaccZ","LLAgyrX","LLAgyrY","LLAgyrZ","LLAmagX","LLAmagY","LLAmagZ","LLAq1","LLAq2","LLAq3","LLAq4",
            "RCaccX","RCaccY","RCaccZ","RCgyrX","RCgyrY","RCgyrZ","RCmagX","RCmagY","RCmagZ","RCq1","RCq2","RCq3","RCq4",
            "RTaccX","RTaccY","RTaccZ","RTgyrX","RTgyrY","RTgyrZ","RTmagX","RTmagY","RTmagZ","RTq1","RTq2","RTq3","RTq4",
            "LTaccX","LTaccY","LTaccZ","LTgyrX","LTgyrY","LTgyrZ","LTmagX","LTmagY","LTmagZ","LTq1","LTq2","LTq3","LTq4",
            "LCaccX","LCaccY","LCaccZ","LCgyrX","LCgyrY","LCgyrZ","LCmagX","LCmagY","LCmagZ","LCq1","LCq2","LCq3","LCq4",
        "label"]
path = "D:/final project/realistic_sensor_displacement/"
# for filename in glob.glob(os.path.join(path, '*.log')):
#     print(filename)
#     data = np.loadtxt(filename)
#     df = pd.DataFrame(data)
#     df.to_csv(filename+".csv", header=header, index=False)



#activity [1, 2, 3, 5, 6, 7, 8, 31, 32, 33]
act = range(1,34)
# act = [1, 2, 3, 5, 6, 7, 8, 31, 32, 33]
path1 = "D:/final project/all_ideal/"
path2 = "D:/final project/all_self/"
# for filename in glob.glob(os.path.join(path, '*ideal.log.csv')):
#     df = pd.read_csv(filename)
#     filename = filename.replace("D:/final project/realistic_sensor_displacement", "")
#     filename = filename.replace(".log.csv", "")
#     print(filename)
#     for i in act:
#         data = df[df["label"] == i]
#         data.to_csv(path1 + filename + str(i) + ".csv", index=False)

#split data
header2 = [
            "RLAaccX","RLAaccY","RLAaccZ","RLAgyrX","RLAgyrY","RLAgyrZ","RLAmagX","RLAmagY","RLAmagZ","RLAq1","RLAq2","RLAq3","RLAq4",
            "RUAaccX","RUAaccY","RUAaccZ","RUAgyrX","RUAgyrY","RUAgyrZ","RUAmagX","RUAmagY","RUAmagZ","RUAq1","RUAq2","RUAq3","RUAq4",
            "BACKaccX","BACKaccY","BACKaccZ","BACKgyrX","BACKgyrY","BACKgyrZ","BACKmagX","BACKmagY","BACKmagZ","BACKq1","BACKq2","BACKq3","BACKq4",
            "LUAaccX","LUAaccY","LUAaccZ","LUAgyrX","LUAgyrY","LUAgyrZ","LUAmagX","LUAmagY","LUAmagZ","LUAq1","LUAq2","LUAq3","LUAq4",
            "LLAaccX","LLAaccY","LLAaccZ","LLAgyrX","LLAgyrY","LLAgyrZ","LLAmagX","LLAmagY","LLAmagZ","LLAq1","LLAq2","LLAq3","LLAq4",
            "RCaccX","RCaccY","RCaccZ","RCgyrX","RCgyrY","RCgyrZ","RCmagX","RCmagY","RCmagZ","RCq1","RCq2","RCq3","RCq4",
            "RTaccX","RTaccY","RTaccZ","RTgyrX","RTgyrY","RTgyrZ","RTmagX","RTmagY","RTmagZ","RTq1","RTq2","RTq3","RTq4",
            "LTaccX","LTaccY","LTaccZ","LTgyrX","LTgyrY","LTgyrZ","LTmagX","LTmagY","LTmagZ","LTq1","LTq2","LTq3","LTq4",
            "LCaccX","LCaccY","LCaccZ","LCgyrX","LCgyrY","LCgyrZ","LCmagX","LCmagY","LCmagZ","LCq1","LCq2","LCq3","LCq4",
        "label"]
for a in act:
    # os.makedirs(path2 + str(a))
    for filename in glob.glob(os.path.join(path2,'*f' + str(a) + '.csv')):
        print(filename)
        index = 1
        df = pd.read_csv(filename, names=header2)
        newdf = pd.DataFrame()
        i = 0
        length = df.shape[0]
        dist = 200 #split 4s
        while(True):
            if((i+1)*dist > length):
                break
            newdf = df[(dist*i) + 1:(dist*(i+1)) + 1]
            newdf.to_csv(path2 + str(a) + "/" + str(index)+".csv", index=False)
            index = index + 1
            i = i + 1
            


