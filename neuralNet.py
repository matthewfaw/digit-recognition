from FileReader import getMNIST
import struct, math
import numpy as np

train_images, train_label, test_images, test_label = getMNIST()
num_train_images = train_label.size
num_test_images = test_label.size

class Network(object):
    def __init__(self, net_shape):
        self.w = init_weights(net_shape) # if l1 = 784 nodes, l2=15 nodes, l3=10, then net_shape = [784,15,10]
        self.b = init_biases(net_shape)
        self.size = len(net_shape)

def init_weights(layers):
    w=dict()
    for i in range(1,len(layers)):
        w[i]=np.asmatrix(np.random.randn(layers[i],layers[i-1]))
    return w

def init_biases(layers):
    b = dict()
    for i in range(1,len(layers)):
        b[i]=np.transpose(np.asmatrix(np.random.randn(layers[i])))
    return b
    
def convert_labels(label):
    #INPUT: labels from MNIST Data set, with values 0-9 in each entry
    #OUTPUT: maxrix of dimensions 10 x size of labels, where an entry t[i,j]=1 if
    #           image j is associated with value i
    t = np.zeros((10,label.size))
    for i in range(0,label.size):
        image_val = label[i]
        t[image_val,i]=1
    return np.transpose(np.asmatrix(t))

def sigmoid(z):
    return 1/(1+np.exp(-z))
    
def der_sigmoid(z):
    return np.multiply(sigmoid(z), 1-sigmoid(z)) 
    
def feedforward(N, x, t):
    w = N.w
    b = N.b
    o = dict()
    o[0]=x
    net = dict()
    # Feed input forward through the network
    for entry in w:
        net[entry]=w[entry]*o[entry-1]+b[entry]
        o[entry]=sigmoid(net[entry])
    #actual output of neural network (gets index of the "most fired" neuron)
    y=np.argmax(o[N.size-1], axis=0) 
    y=np.array(y)[0]
    return net,o,y

def backpropogate(o,t,net,N,y,nabla):
    w=N.w
    b=N.b
    saved_deltas=dict()
    for entry in range(1,N.size):
        i = N.size - entry
        saved_deltas[i]=delta(o,t,net,N,i,saved_deltas)
    for l in range(1,N.size):
        w[l]=w[l]-nabla*saved_deltas[l]*np.transpose(o[l-1])
        b[l]=b[l]-nabla*3*saved_deltas[l]
    N.w=w
    N.b=b
    return N
      
def delta(o,t,net,N,entry,saved_deltas):
    w = N.w
    len_of_net = len(w.keys())+1
    if entry==(len_of_net-1):
        return np.multiply(o[entry]-np.transpose(t), der_sigmoid(net[entry]))
    else:
        return np.multiply(np.transpose(w[entry+1])*saved_deltas[entry+1], der_sigmoid(net[entry]))
    
def execute(layers, nabla, num_epochs, rand_sample_size):
    #INPUT: 
    #       layers: [l1,l2,...,ln]=layers of network
    #       nabla: learning coefficient
    #       num_epochs: number of times to loop through training data
    #       rand_sample_size: number of items to train on in each epoch
    N = Network(layers)

    print 'Train network'
    # count=0
    # i=0
    for ep in range(0,num_epochs):
        
        # Generate array of random numbers-->Use this to choose random pictures from the dataset
        rand_pic_indices = np.random.randint(0,num_train_images,size=rand_sample_size)
        t=convert_labels(train_label)
        for image_num in range(0,num_train_images):
            
            if image_num in rand_pic_indices:
                net,o,y = feedforward(N,np.transpose(np.asmatrix(train_images[:,image_num])),t[image_num,:])
                N = backpropogate(o,t[image_num,:],net,N,y,nabla)
        print 'Epoch number: ',ep,'/',num_epochs
        t=convert_labels(test_label)
        result=np.zeros(num_test_images)
        successes = 0
        for image_num in range(0,num_test_images):
            net,o,y = feedforward(N,np.transpose(np.asmatrix(test_images[:,image_num])),t[image_num,:])
            result[image_num]=y
            if test_label[image_num]==y:
                successes+=1
            if image_num == (num_test_images-1):
                # np.savetxt('result_network.txt', result, delimiter=',')
                #  np.savetxt('expected_result.txt',test_label, delimiter=',')
                print successes,'/',num_test_images



# o_j = output of neuron j
# net_j = weighted sum of inputs to neuron
# w_i,j = weight between neurons i and j

execute([784,40,10],1,100,60000)