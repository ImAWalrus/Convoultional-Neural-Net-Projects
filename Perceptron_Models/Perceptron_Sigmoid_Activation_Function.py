import numpy as np

def sigmoid(x,deriv=False):
   if(deriv==True):
       return(x*(1-x))
   return(1/(1+np.exp(-x)))


x = np.random.random((10,2))

#Add 1's along the diagonal
for i in range(10):
    x = np.insert(x,i*2+i,1)

x = np.reshape(x, ((10,3))) #RESHAPED


y = np.array([[166.0],
[221.0],
[244.0],
[61.0],
[190.0],
[131.0],
[164.0],
[216.0],
[134.0],
[146.0]])

m = max(y) #Find max in y

y = y/m #Normalize data


np.random.seed(1)

syn0 = np.random.random((3,1))-1

for j in xrange(200000):
    l0 = x #Inputs
    l1 = sigmoid(np.dot(l0,syn0)) #Dot product
    l1_error = y - l1
    if(j % 1000) == 0:

        print 'Error:'+str(np.mean(np.abs(l1_error)))
    l1_delta = 0.001*l1_error*sigmoid(l1,deriv=True)

    syn0 += l0.T.dot(l1_delta) #Slope and Intercept (WEIGHTS)

print '\nL1 Output after training:'
print l1
print "\nY:"
print y
