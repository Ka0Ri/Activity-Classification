#pylint: skip-file
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv("D:/final project/wholebody_ideal/3/2.csv")
data = (df["RLAaccX"]).as_matrix()
freq = np.fft.fft(data)
amp = np.log(np.sqrt((freq.real)**2 + (freq.imag)**2))
# gaussian = np.exp(-(np.array([-100, -1, -0.5, 0, 0.5, 1, 2, 100])/2)**2/4) #sigma = 2
# result = np.convolve(amp, gaussian, mode="full")
# print(freq)
x = list(range(1,len(amp)+1))
plt.plot(x, amp)

df = pd.read_csv("D:/final project/wholebody_ideal/1f/2.csv")
data1 = (df["RLAaccX"]).as_matrix()
x1 = list(range(1,len(data1)+1))
plt.plot(x1, data1)
plt.show()