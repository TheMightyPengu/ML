import numpy as np

def create_BigData(n):
    class0 = np.random.uniform(low=0.0, high=0.3, size=(n//2, 2))
    labels0 = np.zeros((n//2,))
    
    class1 = np.random.uniform(low=0.7, high=0.9, size=(n//2, 2))
    labels1 = np.ones((n//2,))

    data = np.concatenate((class0, class1), axis=0)
    labels = np.concatenate((labels0, labels1), axis=0)

    randomize = np.random.permutation(n)
    shuffled_data = data[randomize]
    shuffled_labels = labels[randomize]

    half = n // 2
    Xtrain = shuffled_data[:half]
    Xtest = shuffled_data[half:]
    ytrain = shuffled_labels[:half]
    ytest = shuffled_labels[half:]

    return Xtrain, Xtest, ytrain, ytest


def create_BigData2(n):
    class0 = np.random.uniform(0.0, 0.3, int(n//2))
    labels0 = np.random.uniform(0.0, 0.3, int(n//2))
    labeled_class0 = np.column_stack((class0, labels0))

    class1_1 = np.random.uniform(0.0, 0.3, int(n//4))
    class1_2 = np.random.uniform(0.4, 0.9, int(n//4))
    labels1_1 = np.random.uniform(0.4, 0.9, int(n//4))
    labels1_2 = np.random.uniform(0.0, 0.9, int(n//4))
    labeled_class1 = np.vstack((np.column_stack((class1_1, labels1_1)), np.column_stack((class1_2, labels1_2))))

    data = np.vstack((labeled_class0, labeled_class1))
    labels = np.hstack((np.zeros(int(n//2)), np.ones(n)))

    randomize = np.random.permutation(n)
    data = data[randomize]
    labels = labels[randomize]

    half = int(n//2)
    Xtrain = data[:half]
    Ltrain = labels[:half]
    Xtest = data[half:]
    Ltest = labels[half:]

    return Xtrain, Xtest, Ltrain, Ltest


def create_BigData3(n):
    class0 = np.random.uniform(low=[0.4, 0.4], high=[0.6, 0.6], size=(n//2, 2))
    labels0 = np.zeros((n//2,))

    class1_1 = np.random.uniform(low=[0.0, 0.0], high=[0.9, 0.3], size=(n//8, 2))
    class1_2 = np.random.uniform(low=[0.0, 0.7], high=[0.9, 0.9], size=(n//8, 2))
    class1_3 = np.random.uniform(low=[0.0, 0.0], high=[0.3, 0.9], size=(n//8, 2))
    class1_4 = np.random.uniform(low=[0.7, 0.0], high=[0.9, 0.9], size=(n//8, 2))

    class1 = np.concatenate((class1_1, class1_2, class1_3, class1_4), axis=0)
    labels1 = np.ones((n//2,))

    data = np.concatenate((class0, class1), axis=0)
    labels = np.concatenate((labels0, labels1), axis=0)
    
    randomize = np.random.permutation(n)
    data = data[randomize]
    labels = labels[randomize]

    half = int(n//2)
    Xtrain = data[:half]
    Ltrain = labels[:half]
    Xtest = data[half:]
    Ltest = labels[half:]
    
    return Xtrain, Xtest, Ltrain, Ltest


# def create_BigData4(n):
#     class0_1 = np.random.uniform(low=[0.0, 0.0], high=[0.3, 0.3], size=(n//4, 2))
#     class0_2 = np.random.uniform(low=[0.7, 0.7], high=[0.9, 0.9], size=(n//4, 2))
#     class0 = np.concatenate((class0_1, class0_2), axis=0)
#     labels0 = np.ones((n//2,))

#     class1_1 = np.random.uniform(low=[0.7, 0.0], high=[0.9, 0.3], size=(n//4, 2))
#     class1_2 = np.random.uniform(low=[0.0, 0.7], high=[0.3, 0.9], size=(n//4, 2))
#     class1 = np.concatenate((class1_1, class1_2), axis=0)
#     labels1 = np.zeros((n//2,))

#     data = np.concatenate((class0, class1), axis=0)
#     labels = np.concatenate((labels0, labels1), axis=0)

#     randomize = np.random.permutation(n)
#     data = data[randomize]
#     labels = labels[randomize]

#     half = int(n//2)
#     Xtrain = data[:half]
#     Ltrain = labels[:half]
#     Xtest = data[half:]
#     Ltest = labels[half:]

#     return Xtrain, Xtest, Ltrain, Ltest
def create_BigData4(n):
    class0_1 = np.random.uniform(low=[0.0, 0.0], high=[0.3, 0.3], size=(n // 4, 2))
    class0_2 = np.random.uniform(low=[0.7, 0.7], high=[0.9, 0.9], size=(n // 4, 2))
    class0 = np.concatenate((class0_1, class0_2), axis=0)
    labels0 = np.ones((class0.shape[0],))

    class1_1 = np.random.uniform(low=[0.7, 0.0], high=[0.9, 0.3], size=(n // 4, 2))
    class1_2 = np.random.uniform(low=[0.0, 0.7], high=[0.3, 0.9], size=(n // 4, 2))
    class1 = np.concatenate((class1_1, class1_2), axis=0)
    labels1 = np.zeros((class1.shape[0],))

    data = np.concatenate((class0, class1), axis=0)
    labels = np.concatenate((labels0, labels1), axis=0)

    randomize = np.random.permutation(n)
    data = data[randomize]
    labels = labels[randomize]

    half = int(n // 2)
    Xtrain = data[:half]
    Ltrain = labels[:half]
    Xtest = data[half:]
    Ltest = labels[half:]

    return Xtrain, Xtest, Ltrain, Ltest


def create_BigData5(n):        
    class0 = np.random.uniform(low=0.0, high=0.5, size=(n//2, 2))
    labels0 = np.zeros((n//2,))
    
    class1 = np.random.uniform(low=0.3, high=0.9, size=(n//2, 2))
    labels1 = np.ones((n//2,))

    data = np.concatenate((class0, class1), axis=0)
    labels = np.concatenate((labels0, labels1), axis=0)

    randomize = np.random.permutation(n)
    shuffled_data = data[randomize]
    shuffled_labels = labels[randomize]

    half = n // 2
    Xtrain = shuffled_data[:half]
    Xtest = shuffled_data[half:]
    Ltrain = shuffled_labels[:half]
    Ltest = shuffled_labels[half:]

    return Xtrain, Xtest, Ltrain, Ltest