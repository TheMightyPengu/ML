import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap
import BigData as BD
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle


class Perceptron:
    
    def __init__(self, learning_rate, epohs, threshold = 0):
        self.lr = learning_rate
        self.epohs = epohs
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def plot(self, Xtrain, Ltrain, Xtest, Ltest):
        fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(15, 5))
        # self.epohs = int(self.epohs)

        for i in range(len(Xtest)):
            if Ltest[i] == 0:
                ax1.scatter(Xtest[i][0], Xtest[i][1], marker='o', color='blue', edgecolors='blue', facecolors='none')
            else:
                ax1.scatter(Xtest[i][0], Xtest[i][1], marker='*', color='#FF00FF')
        ax1.set_xlim([-0.015, 1.015])
        ax1.set_ylim([-0.015, 1.015])
        ax1.set_title("Γράφημα προτύπων")
        ax1.set_xlabel("Άξονας Χ")
        ax1.set_ylabel("Άξονας Υ")

        for epoh in range(self.epohs):

            self.train_epoh(Xtrain, Ltrain)
            predictions = self.predict(Xtest)

            ax2.clear()
            ax2.scatter(Xtest[predictions==1, 0], Xtest[predictions==1, 1], c='#FF00FF', marker='*', edgecolors='#FF00FF')
            ax2.scatter(Xtest[predictions==0, 0], Xtest[predictions==0, 1], marker='o', edgecolors='blue', facecolors='none')
            ax2.plot(np.linspace(0, 1, 200), -(self.weights[0]*np.linspace(0, 1, 200) + self.bias) / self.weights[1])
            ax2.set_xlim([-0.015, 1.015])
            ax2.set_ylim([-0.015, 1.015])
            ax2.set_title("Γράφημα προτύπων - Εκαπίδευση" + (f'\n(Epoh {epoh+1})'))
            ax2.set_xlabel("Άξονας Χ")  
            ax2.set_ylabel("Άξονας Υ")

            ax3.clear()
            ax3.scatter(range(len(predictions[Ltest == 0])), predictions[Ltest == 0], marker='*', c='blue', label='y_test == 0')
            ax3.scatter(range(len(predictions[Ltest == 0]), len(Ltest)), predictions[Ltest == 1], marker='*', c='#FF00FF', label='Ltest == 1')
            ax3.set_xlim([-0.3, Xtrain.shape[0]])
            ax3.set_ylim([-0.015, 1.015])
            ax3.set_title("Γράφημα εξόδων προτύπων" + (f' (Epoh {epoh+1})'))
            ax3.set_xlabel("Πρότυπο")
            ax3.set_ylabel("Έξοδος Y")


            plt.draw()
            plt.pause(0.01)
            time.sleep(0.01)

        plt.close(fig)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        for i in range(len(Xtest)):
            if Ltest[i] == 0:
                ax1.scatter(Xtest[i][0], Xtest[i][1], marker='o', color='blue', edgecolors='blue', facecolors='none')
            else:
                ax1.scatter(Xtest[i][0], Xtest[i][1], marker='*', color='#FF00FF')    

        ax1.set_xlim([-0.015, 1.015])
        ax1.set_ylim([-0.015, 1.015])
        ax1.set_title("Γράφημα προτύπων")
        ax1.set_xlabel("Άξονας Χ")
        ax1.set_ylabel("Άξονας Υ")

        ax2.scatter(Xtest[predictions==1, 0], Xtest[predictions==1, 1], c='#FF00FF', marker='*', edgecolors='#FF00FF')
        ax2.scatter(Xtest[predictions==0, 0], Xtest[predictions==0, 1], marker='o', edgecolors='blue', facecolors='none')
        ax2.plot(np.linspace(0, 1, 200), -(self.weights[0]*np.linspace(0, 1, 200) + self.bias) / self.weights[1])
        ax2.set_xlim([-0.015, 1.015])
        ax2.set_ylim([-0.015, 1.015])
        ax2.set_title("Γράφημα προτύπων - Εκαπίδευση" + (f' (Epoh {epoh+1})'))
        ax2.set_xlabel("Άξονας Χ")  
        ax2.set_ylabel("Άξονας Υ")

        ax3.scatter(range(len(predictions[Ltest == 0])), predictions[Ltest == 0], marker='*', c='blue', label='y_test == 0')
        ax3.scatter(range(len(predictions[Ltest == 0]), len(Ltest)), predictions[Ltest == 1], marker='*', c='#FF00FF', label='Ltest == 1')
        ax3.set_xlim([-0.3, Xtrain.shape[0]])
        ax3.set_ylim([-0.015, 1.015])
        ax3.set_title("Γράφημα εξόδων προτύπων" + (f' (Epoh {epoh+1})'))
        ax3.set_xlabel("Πρότυπο")
        ax3.set_ylabel("Έξοδος Y")

        for i in range(len(Xtest)):
            if predictions[i] == 0:
                ax4.scatter(i+1, 0, marker='o',  color='blue', facecolors='none', s=150)
            else:
                ax4.scatter(i+1, 1, marker='o', color='#FF00FF', facecolors='none', s=150)
        ax4.scatter(np.where(Ltest == 0)[0]+1, Ltest[Ltest == 0], marker='x', c='#00FF00')
        ax4.scatter(np.where(Ltest == 1)[0]+1, Ltest[Ltest == 1], marker='x', c='#00FF00')
        ax4.set_xlim([0, len(Xtest)+0.5])
        ax4.set_ylim([-0.02,  1.02])
        ax4.set_title("Γράφημα εξόδων-στόχων προτύπων")
        ax4.set_xlabel("Πρότυπο")
        ax4.set_ylabel("Έξοδος Y")
        markers = [
                    Line2D([0], [0], marker=MarkerStyle(marker='o', fillstyle='none'), color='blue', label='Class 0 prediction', linestyle=''),
                    Line2D([0], [0], marker=MarkerStyle(marker='o', fillstyle='none'), color='#FF00FF', label='Class 1 prediction', linestyle=''),
                    Line2D([0], [0], marker='x', color='#00FF00', label='Where they should be', linestyle='')
                  ]
        ax4.legend(handles=markers, loc='center left')

        plt.show()


    def train_epoh(self, X, y):
        if self.weights is None:
            self.weights = np.random.rand(X.shape[1])
        if self.bias is None:
            self.bias = np.random.rand()

        for i in range(X.shape[0]):
            linear_output = np.dot(X[i], self.weights) + self.bias
            predictions = self.activation(linear_output)
            update = self.lr * (y[i] - predictions)
            self.weights += update * X[i]
            self.bias += update
                
    def predict(self, X):
        weighted_sum = np.dot(X, self.weights) + self.bias
        predictions = self.activation(weighted_sum)
        return predictions

while True:
    n = input("Please give a number that's a multiple of 8: ")
    if n.isdigit() and int(n) % 8 == 0:
        n = int(n)
        break

print('''
Your options are:
1. Γραμμικά Διαχωρίσιμα Πρότυπα
2. Μη Γραμμικά Διαχωρίσιμα Πρότυπα – Κλάση 0 στη Γωνία
3. Μη Γραμμικά Διαχωρίσιμα Πρότυπα - Κλάση 0 στο Κέντρο
4. Μη Γραμμικά Διαχωρίσιμα Πρότυπα - Πύλη XOR
5. Μη Γραμμικά Διαχωρίσιμα Πρότυπα - Κλάση 0 μέσα στην Κλάση 1
6. Τέλος
''')

while True:
    option = input("\nPlease pick an option: ")
    if option.isdigit():
        option = int(option)

    if option == 1:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData(n)
        perceptron = Perceptron(learning_rate, epohs)
        perceptron.plot(Xtrain, Ltrain, Xtest, Ltest)


    elif option == 2:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData2(n)
        perceptron = Perceptron(learning_rate, epohs)
        perceptron.plot(Xtrain, Ltrain, Xtest, Ltest)


    elif option == 3:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData3(n)
        perceptron = Perceptron(learning_rate, epohs)
        perceptron.plot(Xtrain, Ltrain, Xtest, Ltest)


    elif option == 4:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData4(n)
        perceptron = Perceptron(learning_rate, epohs)
        perceptron.plot(Xtrain, Ltrain, Xtest, Ltest)


    elif option == 5:

        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData5(n)
        perceptron = Perceptron(learning_rate, epohs)
        perceptron.plot(Xtrain, Ltrain, Xtest, Ltest)


    elif option == 6:
        print("Thank you for visiting, please come again.")
        break
    
    else:
        print("Thats not an option :(")