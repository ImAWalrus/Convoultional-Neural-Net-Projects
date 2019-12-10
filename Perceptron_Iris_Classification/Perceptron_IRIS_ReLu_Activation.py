import csv
import numpy as np

def ReLu(x,deriv=False):
    x_cp = np.copy(x)
    if(deriv==True):
        x_cp[x_cp < 0] = 0
        x_cp[x_cp >= 0] = 1
        return x_cp
    return np.maximum(0.1*x_cp,x_cp)


def iris_CSV():
    #################X Inputs####################
    fp = open('iris_data.dat','r')
    data = csv.reader(fp,delimiter=',')

    x = []

    for i in data:
        x.append(i[:4])

    fp.close()
    #################Y Outputs####################
    fp = open('iris_data.dat','r')
    data = csv.reader(fp,delimiter=',')

    y = []

    for i in data:
        y.append(i[-1:])

    fp.close()
    #################Y Classification####################
    y_bin = [];
    for i in y:
        for j in i:
            if j == 'setosa':
                y_bin.append((1,0,0))
            elif j == 'versicolor':
                y_bin.append((0,1,0))
            elif j == 'virginica':
                y_bin.append((0,0,1))

    return x, y_bin


def main():
    a,b = iris_CSV()

    x = np.array(a)
    x = x.astype(float)

    y = np.array(b)

    print x.shape
    print y.shape

    np.random.seed(1)

    syn0 = 2*np.random.random((4,5))-1
    syn1 = 2*np.random.random((5,3))-1

    for j in xrange(50000):
        l0 = x
        l1 = ReLu(np.dot(l0,syn0))
        l2 = ReLu(np.dot(l1,syn1))
        l2_error = y - l2


        if(j % 1000) == 0:
            print('Iteration {}: | Error:{}'.format(j,str(np.mean(np.abs(l2_error)))))
        l2_delta = l2_error*ReLu(l2,deriv=True)
        l1_error = l2_delta.dot(syn1.T)
        l1_delta = l1_error*ReLu(l1,deriv=True)


        syn1 += 0.0001*l1.T.dot(l2_delta)
        syn0 += 0.0001*l0.T.dot(l1_delta)


    print '\nL2 Output after training:'
    print l2
    print '\nY Output'
    print y

if __name__ == '__main__':
    main()
