import pickle as pkl

from matplotlib import pyplot as plt
import numpy as np

def one_hot(labels, num_classes):
    """
    Return:
       Labels in one hot form
    """
    
    return np.squeeze(np.eye(num_classes)[labels.reshape(-1)])



def get_mnist_dataset():
    """
    Return:
        Mnist data set in flatten form
    """
    with open('knn/mnist.pkl', 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        train, val, test = u.load()
         
    train_images, train_labels = train
    test_images, test_labels = test   
    
    #train_labels, test_labels = one_hot(train_labels, num_classes=10), one_hot(test_labels, num_classes=10)
    
    train = (train_images, train_labels)
    test = (test_images, test_labels)
        
    return train, test

    
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    plt.imshow(image, cmap='gray')
    plt.show()
