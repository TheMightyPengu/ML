import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import ListedColormap
import BigData as BD

class Adaline:
    
    def __init__(self, learning_rate, epohs, threshold = 0):
        self.lr = learning_rate
        self.epohs = epohs
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.trained = 0

    def activation(self, x):
        return x
    
    def weighted_sum(self, X):
        weighted_sum = np.dot(X, self.weights) + self.bias
        return weighted_sum
    
    def fit_plot(self, Xtrain, Ltrain, Xtest, Ltest):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6))

        # self.epohs = int(self.epohs)

        for i in range(len(Xtest)):
            if Ltest[i] == 0:
                ax1.scatter(Xtest[i][0], Xtest[i][1], marker='o', color='blue', edgecolors='blue', facecolors='none')
            else:
                ax1.scatter(Xtest[i][0], Xtest[i][1], marker='*', color='#FF00FF')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_title("Γράφημα προτύπων")
        ax1.set_xlabel("Άξονας Χ")
        ax1.set_ylabel("Άξονας Υ")

        for epoh in range(self.epohs):

            self.train_epoh(Xtrain, Ltrain)
            predictions = self.predict(Xtest)

            ax2.clear()
            ax2.scatter(Xtest[predictions==1, 0], Xtest[predictions==1, 1], c='#FF00FF', marker='*', edgecolors='#FF00FF')
            ax2.scatter(Xtest[predictions==0, 0], Xtest[predictions==0, 1], marker='o', edgecolors='blue', facecolors='none')

            w1, w2 = self.weights
            fix = -(w1 * (np.linspace(0, 1, 200) - np.mean(Xtest[:, 0])) / np.std(Xtest[:, 0]) + self.bias) / w2
            fix = fix * np.std(Xtest[:, 0]) + np.mean(Xtest[:, 0])

            ax2.plot(np.linspace(0, 1, 200), fix)
            ax2.set_xlim([0, 1])
            ax2.set_ylim([0, 1])
            ax2.set_title("Γράφημα προτύπων - Εκαπίδευση" + (f'\n(Epoh {epoh+1})'))
            ax2.set_xlabel("Άξονας Χ")  
            ax2.set_ylabel("Άξονας Υ")

            ax3.clear()
            outputs = self.predict(Xtrain)
            ax3.scatter(range(Xtrain.shape[0]), outputs, c=Ltrain, cmap=ListedColormap(['blue', '#FF00FF']), marker='*')
            ax3.set_xlim([0, Xtrain.shape[0]])
            ax3.set_ylim([0, 1])
            ax3.set_title("Γράφημα εξόδων προτύπων" + (f'\n(Epoh {epoh+1})'))
            ax3.set_xlabel("Πρότυπο")
            ax3.set_ylabel("Έξοδος Y")


            plt.draw()
            plt.pause(0.01)
            time.sleep(0.01)

        plt.close(fig)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 11))

        for i in range(len(Xtest)):
            if Ltest[i] == 0:
                ax1.scatter(Xtest[i][0], Xtest[i][1], marker='o', color='blue', edgecolors='blue', facecolors='none')
            else:
                ax1.scatter(Xtest[i][0], Xtest[i][1], marker='*', color='#FF00FF')
                
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_title("Γράφημα προτύπων")
        ax1.set_xlabel("Άξονας Χ")
        ax1.set_ylabel("Άξονας Υ")

        ax2.scatter(Xtest[predictions==1, 0], Xtest[predictions==1, 1], c='#FF00FF', marker='*', edgecolors='#FF00FF')
        ax2.scatter(Xtest[predictions==0, 0], Xtest[predictions==0, 1], marker='o', edgecolors='blue', facecolors='none')
        
        w1, w2 = self.weights
        fix = -(w1 * (np.linspace(0, 1, 200) - np.mean(Xtest[:, 0])) / np.std(Xtest[:, 0]) + self.bias) / w2
        fix = fix * np.std(Xtest[:, 0]) + np.mean(Xtest[:, 0])

        ax2.plot(np.linspace(0, 1, 200), fix)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1])
        ax2.set_title("Γράφημα προτύπων - Εκαπίδευση" + (f' (Epoh {epoh+1})'))
        ax2.set_xlabel("Άξονας Χ")  
        ax2.set_ylabel("Άξονας Υ")

        ax3.scatter(range(Xtrain.shape[0]), outputs, c=Ltrain, cmap=ListedColormap(['blue', '#FF00FF']), marker='*')
        ax3.set_xlim([0, Xtrain.shape[0]])
        ax3.set_ylim([0, 1])
        ax3.set_title("Γράφημα εξόδων προτύπων" + (f' (Epoh {epoh+1})'))
        ax3.set_xlabel("Πρότυπο")
        ax3.set_ylabel("Έξοδος Y")

        correct = Ltest == predictions
        wrong = Ltest != predictions
        ax4.scatter(Xtest[correct, 0], Xtest[correct, 1], marker='o', color='#00FF00', label='Correct')
        ax4.scatter(Xtest[wrong, 0], Xtest[wrong, 1], marker='x', color='red', label='Incorrect')
        ax4.set_xlim([0, 1])
        ax4.set_ylim([0, 1])
        ax4.set_title("Γράφημα εξόδων-στόχων προτύπων")
        ax4.set_xlabel("Πρότυπο")
        ax4.set_ylabel("Έξοδος Y")

        plt.show()

    def train_epoh(self, X, y):
        self.trained += 1
        if self.weights is None:
            self.weights = np.random.rand(X.shape[1])
        if self.bias is None:
            self.bias = 0.0
        cost = 0.0
        for xi, target in zip(X, y):
                sum = self.weighted_sum(xi)
                output = self.activation(sum)
                error = (target - output)
                self.weights += self.lr * xi * error
                self.bias += self.lr * error
                cost = 0.5 * error ** 2
        return cost

    def predict(self, X):
            return np.where(self.activation(self.weighted_sum(X)) >= 0.5, 1, 0)






# while True:
#     n = input("Please give a number that's a multiple of 8: ")
#     if n.isdigit() and int(n) % 8 == 0:
#         n = int(n)
#         break
n = 40

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
        adaline = Adaline(learning_rate, epohs)
        adaline.fit_plot(Xtrain, Ltrain, Xtest, Ltest)


    elif option == 2:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData2(n)
        adaline = Adaline(learning_rate, epohs)
        adaline.fit_plot(Xtrain, Ltrain, Xtest, Ltest)


    elif option == 3:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData3(n)
        adaline = Adaline(learning_rate, epohs)
        adaline.fit_plot(Xtrain, Ltrain, Xtest, Ltest)


    elif option == 4:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData4(n)
        adaline = Adaline(learning_rate, epohs)
        adaline.fit_plot(Xtrain, Ltrain, Xtest, Ltest)


    elif option == 5:

        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData5(n)
        adaline = Adaline(learning_rate, epohs)
        adaline.fit_plot(Xtrain, Ltrain, Xtest, Ltest)


    elif option == 6:
        print("Thank you for visiting, please come again.")
        break

    else:
        print("Thats not an option :(")