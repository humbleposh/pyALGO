#import numpy module
#import numpy
import numpy as np
#import ESN module
from pyESN import ESN
#import matplotlib
import matplotlib.pyplot as plt

#read the open, high, low and closing price from the csv files
o, h, l, c=np.loadtxt('C:/AAA-TEMP/PyALGO/GBPUSD60.csv', delimiter=',',
                      usecols=(2,3,4,5), unpack=True)
# o, h, l, c=np.loadtxt('C:/AAA-TEMP/PyALGO/GBPUSD1H-20160401-20170331.csv', delimiter=',',
#                      usecols=(2,3,4,5), unpack=True)

##build an Echo State Network
esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = 8000,
          spectral_radius = 1.5,
          random_state=42)
#choose the training set
trainlen = 500
future = 20
#start training the model
pred_training = esn.fit(np.ones(trainlen),c[len(c)-trainlen: len(c)])
#make the predictions
prediction = esn.predict(np.ones(future))
print("test error: \n"+str(np.sqrt(np.mean((prediction.flatten() \
- c[trainlen:trainlen+future])**2))))

#print the predicted values of the closing price
prediction
#plot the predictions
plt.figure(figsize=(11,1.5))
plt.plot(range(0,trainlen),c[len(c)-trainlen:len(c)],'k',label="target system")
plt.plot(range(trainlen,trainlen+future),prediction,'r', label="free running ESN")
lo,hi = plt.ylim()
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1.1),fontsize='x-small')
plt.show()