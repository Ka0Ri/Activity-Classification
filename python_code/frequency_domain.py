#pylint: skip-file
import pandas as pd 
import numpy as np 
import os
act = [1, 2, 3, 5, 6, 7, 8, 31, 32, 33]
path1 = "D:/final project/wholebody_ideal/"
path2 = "D:/final project/wholebody_self/"
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
    for filename in os.listdir(path2 + str(a)):
        print(filename)
        df = pd.read_csv(path2 + str(a) + '/' + filename)
        matrix = df.as_matrix()
        frequencydata = np.zeros(np.shape(matrix))
        for i in range(0, 117):
            data = matrix[:,i]
            freq = np.fft.fft(data)
            frequencydata[:,i] = np.log(np.sqrt((freq.real)**2 + (freq.imag)**2))
        frequencydata[:,117] = matrix[:,117]
        df = pd.DataFrame(frequencydata)
        df.to_csv(path2 + str(a) + "f/" + filename,header=header2, index=False)