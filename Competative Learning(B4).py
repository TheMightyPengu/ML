import numpy as np
import matplotlib.pyplot as plt
import BigData as BD
import sys
from tkinter import messagebox
from sklearn.cluster import KMeans

class Competitive_Learning:
    def __init__(self, nebrones, lr):
        self.nebrones = nebrones
        self.w = np.random.random_sample((self.nebrones, x.shape[1]))
        self.lr = lr

    def partial_fit(self, x, epohs, current_epoh):
        norms = np.linalg.norm(self.w - np.expand_dims(x, 0), axis=-1)
        indx = norms.argmin()
        self.w[indx] += self.lr * (x - self.w[indx])
        self.lr *= (1 - (current_epoh / epohs))

    def plot2(self, i):
        plt.close()
        fig, ax1 = plt.subplots()
        ax1.set_xlim([-0.015, 1.015])
        ax1.set_ylim([-0.015, 1.015])
        ax1.set_xlabel("Άξονας Χ")
        ax1.set_ylabel("Άξονας Υ")
        ax1.set_title("Γράφημα προτύπων - Συνάψεων (Epoch: {})".format(i))
        ax1.scatter(self.w[:, 0], self.w[:, 1], marker='o', color='black', edgecolors='black', s=100)
        for k in range(len(Xtest)):
            if Ltest[k] == 0:
                ax1.scatter(Xtest[k][0], Xtest[k][1], marker='o', color='blue', edgecolors='blue', facecolors='none')
            else:
                ax1.scatter(Xtest[k][0], Xtest[k][1], marker='*', color='#FF00FF')

    def plot(self, Xtrain, Xtest, Ltest, epohs, training=False): #THIS IS THE TRUE COLORING
        fig, ax1 = plt.subplots()
        ax1.set_xlim([-0.015, 1.015])
        ax1.set_ylim([-0.015, 1.015])
        ax1.set_xlabel("Άξονας Χ")
        ax1.set_ylabel("Άξονας Υ")

        prev_results = None

        for i in range(1, epohs + 1):
            current_results = np.copy(self.w)

            for j, x in enumerate(Xtrain):
                if training:
                    self.partial_fit(x, epohs, i)

            ax1.cla()
            ax1.scatter(self.w[:, 0], self.w[:, 1], marker='o', color='black', edgecolors='black', s=100)

            for k in range(len(Xtest)):
                if Ltest[k] == 0:
                    ax1.scatter(Xtest[k][0], Xtest[k][1], marker='o', color='blue', edgecolors='blue', facecolors='none')
                else:
                    ax1.scatter(Xtest[k][0], Xtest[k][1], marker='*', color='#FF00FF')

            ax1.set_title("Γράφημα προτύπων - Συνάψεων (Epoch: {})".format(i))
            plt.pause(0.01)

            if prev_results is not None and np.all(current_results == prev_results):
                messagebox.showinfo(title="Training finished", message=f'The training has been finished because the results have been the same for two consecutive epohs, here are the results.')
                self.plot2(i)
                break

            prev_results = current_results 

        plt.show()

    # def plot(self, Xtrain, Xtest, Ltest, epohs, training=False): #THIS IS THE CLUSTERING COLORING
    #     fig, ax1 = plt.subplots()
    #     ax1.set_xlim([-0.015, 1.015])
    #     ax1.set_ylim([-0.015, 1.015])
    #     ax1.set_xlabel("Άξονας Χ")
    #     ax1.set_ylabel("Άξονας Υ")

    #     prev_results = None

    #     for i in range(1, epohs + 1):
    #         current_results = np.copy(self.w)

    #         for j, x in enumerate(Xtrain):
    #             if training:
    #                 self.partial_fit(x, epohs, i)

    #         ax1.cla()
    #         ax1.scatter(self.w[:, 0], self.w[:, 1], marker='o', color='black', edgecolors='black', s=100)

    #         kmeans = KMeans(n_clusters=self.nebrones, init=self.w, n_init=1, max_iter=1)
    #         kmeans.fit(Xtest)
    #         cluster_labels = kmeans.labels_

    #         unique_labels = np.unique(cluster_labels)
    #         colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'brown', 'pink', 'gray', 'olive']
    #         color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    #         for k in range(len(Xtest)):
    #             prediction = cluster_labels[k]
    #             marker_color = color_map[prediction]
    #             ax1.scatter(Xtest[k][0], Xtest[k][1], marker='o', color=marker_color, edgecolors=marker_color, facecolors='none')

    #         ax1.set_title("Γράφημα προτύπων - Συνάψεων (Epoch: {})".format(i))
    #         plt.pause(0.01)

    #         if prev_results is not None and np.all(current_results == prev_results):
    #             messagebox.showinfo(title="Training finished", message=f'The training has been finished because the results have been the same for two consecutive epohs, here are the results.')
    #             self.plot2(i)
    #             break

    #         prev_results = current_results

    #     plt.show()

    # def plot(self, Xtrain, Xtest, Ltest, epohs, training=False):   # this is prediction coloring
    #     fig, ax1 = plt.subplots()
    #     ax1.set_xlim([-0.015, 1.015])
    #     ax1.set_ylim([-0.015, 1.015])
    #     ax1.set_xlabel("Άξονας Χ")
    #     ax1.set_ylabel("Άξονας Υ")

    #     prev_results = None

    #     for i in range(1, epohs + 1):
    #         current_results = np.copy(self.w)

    #         for j, x in enumerate(Xtrain):
    #             if training:
    #                 self.partial_fit(x, epohs, i)

    #         ax1.cla()
    #         ax1.scatter(self.w[:, 0], self.w[:, 1], marker='o', color='black', edgecolors='black', s=100)

    #         for k in range(len(Xtest)):
    #             prediction = np.argmin(np.linalg.norm(self.w - Xtest[k], axis=1))
    #             if prediction == 0:
    #                 ax1.scatter(Xtest[k][0], Xtest[k][1], marker='o', color='blue', edgecolors='blue', facecolors='none')
    #             else:
    #                 ax1.scatter(Xtest[k][0], Xtest[k][1], marker='*', color='#FF00FF')

    #         ax1.set_title("Γράφημα προτύπων - Συνάψεων (Epoch: {})".format(i))
    #         plt.pause(0.01)

    #         if prev_results is not None and np.all(current_results == prev_results):
    #             messagebox.showinfo(title="Training finished", message=f'The training has been finished because the results have been the same for two consecutive epohs, here are the results.')
    #             self.plot2(i)
    #             break

    #         prev_results = current_results 

    #     plt.show()

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

lr = 0.5
nebrones = 4
epohs = 100
x,y=None,None

while True:
    option = input("\nPlease pick an option: ")
    if option.isdigit():
        option = int(option)

    if option == 1:
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData(n)
        x = Xtrain
        y = Ltrain
        lr = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        nebrones = int(input('Please pick the number of neurons you want the algorith to use: '))        
        model = Competitive_Learning(nebrones, lr)
        model.plot(Xtrain, Xtest, Ltest, epohs, True)

    elif option == 2:
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData2(n)
        x = Xtrain
        y = Ltrain
        lr = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        nebrones = int(input('Please pick the number of neurons you want the algorith to use: '))    
        model = Competitive_Learning(nebrones, lr)
        model.plot(Xtrain, Xtest, Ltest, epohs, True)

    elif option == 3:
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData3(n)
        x = Xtrain
        y = Ltrain
        lr = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        nebrones = int(input('Please pick the number of neurons you want the algorith to use: '))    
        model = Competitive_Learning(nebrones, lr)
        model.plot(Xtrain, Xtest, Ltest, epohs, True)

    elif option == 4:
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData4(n)
        x = Xtrain
        y = Ltrain
        lr = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        nebrones = int(input('Please pick the number of neurons you want the algorith to use: '))    
        model = Competitive_Learning(nebrones, lr)
        model.plot(Xtrain, Xtest, Ltest, epohs, True)

    elif option == 5:
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData5(n)
        x = Xtrain
        y = Ltrain
        lr = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        nebrones = int(input('Please pick the number of neurons you want the algorith to use: '))    
        model = Competitive_Learning(nebrones, lr)
        model.plot(Xtrain, Xtest, Ltest, epohs, True)

    elif option == 6:
        print("Thank you for visiting, please come again.")
        sys.exit()