import csv
import numpy as np

def identity(x,deriv=False):
    if deriv:
        return 1
    return x


def iris_CSV():
    #################X Inputs####################
    fp = open('pima.txt','r')
    data = csv.reader(fp,delimiter=',')

    x = []

    for i in data:
        x.append(i[:8])



    fp.close()

    #################Y Outputs####################
    fp = open('pima.txt','r')
    data = csv.reader(fp,delimiter=',')

    y = []

    for i in data:
        y.append(i[-1:])

    fp.close()


    return x, y

def main():
    a,b = iris_CSV()

    x = np.array(a)
    x = x.astype(float)

    y = np.array(b)
    y = y.astype(float)

    max_1 = np.max(x,axis = 0)
    min_1 = np.min(x,axis=0)
    x = (x-min_1)/(max_1-min_1)


    print x.shape
    print y.shape

    np.random.seed(1)

    for i in range(6,11):
        print("\nCurrent Hidden Layer :{}\n".format(i))

        syn0 = 2*np.random.random((8,i))-1
        syn1 = 2*np.random.random((i,1))-1

        #PLOT
        for j in xrange(100000):
            l0 = x
            l1 = identity(np.dot(l0,syn0))
            l2 = identity(np.dot(l1,syn1))
            l2_error = y - l2


            if(j % 10000) == 0:
                print('Iteration {}: | Error:{}'.format(j,str(np.mean(np.abs(l2_error)))))
            l2_delta = 0.001*l2_error*identity(l2,deriv=True)
            l1_error = l2_delta.dot(syn1.T)
            l1_delta = 0.001*l1_error*identity(l1,deriv=True)


            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)


        print '\nL2 Output after training:'
        print l2
        print '\nY Output'
        print y

if __name__ == '__main__':
    main()
