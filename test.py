import numpy as np
import matplotlib.pyplot as plt


class Hopfield:
    def __init__(self, type):
        self.style(type)

    @staticmethod
    def __discrete(x):
        return -1 if x < 0 else 1
    
    @staticmethod
    def make_weight(vector):
        num_neurons = len(vector[0])
        result = np.zeros((num_neurons, num_neurons))
        for v in vector:
            result += np.outer(v, v)
        np.fill_diagonal(result, 0)
        return result


    def style(self, type: str):
        if type == 'async':
            self.ann = self.__r_async
        elif type == 'sync':
            self.ann = self.__r_sync
        else:
            raise ValueError('Not a valid type')

    def fit(self, data):
        self.w = self.make_weight(data)

    def test(self, data, label, max_iters=10):
        for x, y in zip(data, label):
            output, iters = self.ann(x, y, max_iters)
            self.plot_images(x, y, output, iters)

    def __r_sync(self, data, label, max_iters):
        col, row = np.shape(self.w)
        output = np.zeros(col)
        temp = data.copy()
        for i in range(1, max_iters + 1):
            for k in range(col):
                if np.any(self.w[k,] != 0.0):
                    temp[k] = self.__discrete(np.matmul(self.w[k,], temp))
            if np.array_equal(temp, output):
                break
            output = temp
        return output, i

    def __r_async(self, data, label, max_iters):
        col, row = np.shape(self.w)
        output = np.zeros(col)
        temp = data.copy()
        for i in range(1, max_iters + 1):
            for k in np.random.permutation(col):
                if np.any(self.w[k,] != 0.0):
                    temp[k] = self.__discrete(np.matmul(self.w[k,], temp))
            if np.array_equal(temp, output):
                break
            output = temp
        return output, i

    @staticmethod
    def plot_images(input_data, input_label, output_data, iterations):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(input_data.reshape(7, 11), cmap=plt.cm.gray)
        plt.title('Input: {}'.format(input_label))
        plt.subplot(1, 2, 2)
        plt.imshow(output_data.reshape(7, 11), cmap=plt.cm.gray)
        plt.title('Output after {} iterations'.format(iterations))
        plt.show()


def run():
    dimensions = 7
    num_neurons = dimensions * dimensions

    hpf = Hopfield('async')

    perfect_patterns = [
        np.array([1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1]),
        np.array([1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1]),
        np.array([1, -1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, -1, -1, 1, 1, 1]),
        np.array([1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1])
    ]

    max_pattern_length = max(len(pattern) for pattern in perfect_patterns)
    perfect_patterns = [np.pad(pattern, (0, max_pattern_length - len(pattern)), mode='constant') for pattern in perfect_patterns]

    hpf.fit(perfect_patterns)

    num_imperfect_patterns = 40  # Change the number of imperfect patterns here
    imperfect_patterns = []
    imperfect_labels = []
    for i in range(num_imperfect_patterns):
        pattern = np.random.choice([-1, 1], size=num_neurons)
        imperfect_patterns.append(pattern)
        imperfect_labels.append(i + 1)

    hpf.test(imperfect_patterns, imperfect_labels)


run()
