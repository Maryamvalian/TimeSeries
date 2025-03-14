import numpy as np
import matplotlib.pyplot as plt
import csv
#
#    Part 1: generate sine data with a,f,fs, size (amplitude,frequency, sample-frequency, size) and add noise
#    part 2: read data from csv file
#    part 3: cross-correlation function where cross correlation_X,Y[k]=sigma(X[n].Y[n+k])

#-----------------------------------------------------------------
def generate_sin(size, a, f, fs):
    n = np.arange(size)            #sample numbers [0,12,..,999] if size=1000
    data = a * np.sin(2 * np.pi * f * n / fs)
    return  data
#----------------------------------------------------
def add_noise(data, mean=0, std=1):
    noise = np.random.normal(mean, std, size=len(data))
    return data + noise

#----------------------------------------
def read_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            data.append(float(row[1]))  #second column of csv
    return np.array(data)

#-------------------------------------------------------------------------
def crosscorrelation(x, y):

    x = np.array(x)
    y = np.array(y)
    
    #subtract the mean
    x = np.array(x) - np.mean(x)
    y = np.array(y) - np.mean(y)
    size = len(x)
    correlation = []     #empty list
    for lag in range(-size + 1, size):
        # Shift y by the current lag
        if lag < 0:
            shifted_y = y[-lag:]  #slice  right
            shifted_x = x[:lag]  # slice left
        elif lag > 0:
            shifted_y = y[:-lag]  # slice left
            shifted_x = x[lag:]  # slice right
        else:
            shifted_y = y  # No shift
            shifted_x = x
        cross_corr = np.sum(shifted_x * shifted_y)
        correlation.append(cross_corr)


    correlation = np.array(correlation) #conver list to array


    lags = np.arange(-size + 1, size)
    plt.plot(lags, correlation)
    plt.grid()
    plt.show()

# -------------------------Main------------------

#part 1

data = generate_sin(1000, 2.0, 5.0, 100.0)
noisy_data = add_noise(data, mean=0, std=1)
plt.plot(data)
plt.plot(noisy_data, color='orange')
plt.show()

#part 3
crosscorrelation(data, noisy_data)

#part 2
data2 = read_data("sp500_80_92.csv")
noisy_data2=add_noise(data2,mean=0,std=1)
plt.plot(data2,color='blue')
plt.plot(noisy_data2,color='orange')
plt.xlim(0,1000)
plt.ylim(50,200)
plt.show()

#part 3
crosscorrelation(data2, data2)




