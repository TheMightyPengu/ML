from turtle import update
import numpy as np
import lines as l
import matplotlib.pyplot as plt
import BigData as BD
from minisom import MiniSom
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib.lines import Line2D 



# class som :
#     def __init__(self,nebrones:int):
#         self.lr=1
#         self.gitonia=1
#         self.nebrones=nebrones
#         self.w = np.random.random_sample((som.nebrones,))

#     def winner(self, X):
#         return np.unravel_index(np.argmin(X,axis=None), X.shape)

#     def ind(self, X):
#         return np.sum(np.square(self.w-X),axis=2)
    
#     def __updw(self,xtr,wnr):
#         x,y=wnr
#         nrs=[]
#         if self.gitonia< 1e-3:
#             self.w[x,y,:]+=self.lr*(xtr-self.w[x,y,:])
#             return nrs 
#         step=int(self.gitonia*10)
      
#         for i in range(max(0,x-step) , min(self.w.shape[0],x+step)):
#              for j in range(max(0,y-step) , min(self.w.shape[1],y+step)):
#                  if((i,j)!=(x,y)):nrs.append((i,j))
#                  nrm=np.exp(-(np.square(i-x)+np.square(j-y))/2/self.gitonia)
#                  self.w[i,j,:]+=self.lr*nrm*(xtr-self.w[i,j,:])
         
#         return nrs
    
#     def fit_nebrones_ohonen(self,x,epohs=5000):
#         geitonia=500
#         self.__init_components(x,False)
#         lr_old=self.lr
#         gitonia_old=self.gitonia
#         self.r=0
#         data=np.copy(x)
#         for e in range(epohs):
#             np.random.shuffle(data)
#             for x in data:
#                 i,j=self._find_winner(
#                     self._ind(x)
#                 )
#                 self.__updw(x,(i,j))
#             self.lr=lr_old*(1-e/epohs) 
#             self.gitonia=gitonia_old*(1-e/geitonia)



###################################################################################

# def plot_animated(som, Xtrain, Ltrain, Xtest, Ltest, ax, fig, num_iteration, input_len, sigma, k1, k2, title=None, callback=None):
#     som.sigma = sigma
#     som.input_len = input_len
#     x = k1  # Replace with the desired value for x
#     y = k2  # Replace with the desired value for y
    
#     ax.set_xlim([-0.2, 1])
#     ax.set_ylim([-0.2, 1])
#     title = ax.text(0.2, 0.85, "", bbox={'facecolor': 'w', 'alpha': 0.5, 'pad': 5},
#                     transform=ax.transAxes, ha="center")
#     epohs = num_iteration
#     scatters = []

#     som_plot = MiniSom(x, y, input_len, sigma, learning_rate)  # Create som_plot with updated parameters

#     def update(epoh):
#         for x in Xtrain:
#             som_plot.update(x, som_plot.winner(x), epoh, epohs)

#         winning_neurons_train = np.array([som_plot.winner(x) for x in Xtrain])
#         winning_neurons_test = np.array([som_plot.winner(x) for x in Xtest])

#         predicted_labels_train = np.zeros(Xtrain.shape[0])
#         for i, (x, neuron) in enumerate(zip(Xtrain, winning_neurons_train)):
#             predicted_labels_train[i] = Ltrain[
#                 np.logical_and(winning_neurons_train[:, 0] == neuron[0], winning_neurons_train[:, 1] == neuron[1])][0]

#         predicted_labels_test = np.zeros(Xtest.shape[0])
#         for i, (x, neuron) in enumerate(zip(Xtest, winning_neurons_test)):
#             predicted_labels_test[i] = Ltrain[
#                 np.logical_and(winning_neurons_train[:, 0] == neuron[0], winning_neurons_train[:, 1] == neuron[1])][0]

#         if scatters:
#             for scatter in scatters:
#                 scatter.remove()
#         scatters.clear()

#         scatter1 = ax.scatter(Xtest[:, 0], Xtest[:, 1], c=predicted_labels_test, cmap='coolwarm',
#                               label='Predicted Labels (Test)')
#         scatters.append(scatter1)

#         centroids = som_plot.get_weights()
#         colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
#         for i, centroid in enumerate(centroids):
#             color = colors[i % len(colors)]
#             scatter = ax.scatter(centroid[:, 0], centroid[:, 1], marker='s', s=50, linewidths=10, color=color)
#             scatters.append(scatter)

#         title.set_text(f'epoh: {epoh + 1}/{epohs}')

#         return scatters + [title]
    
#     epohs = int(epohs)
    
#     anim = FuncAnimation(fig, update, frames=epohs, blit=True, interval=10, repeat=False)

#     return anim

def animation(som, Xtrain, Ltrain, Xtest, Ltest, ax, fig, num_iteration, title=None, active=False, callback=None):
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

    ax2.set_title('Γραφική παράσταση ανταγωνιστικών νευρώνων')
    ax2.set_xlabel("Άξονας Χ")
    ax2.set_ylabel("Άξονας Υ")
    title2 = ax2.text(0.88, 0.87, "", bbox={'facecolor': 'w', 'alpha': 0.4, 'pad': 3}, transform=ax2.transAxes, ha="center")




    def update(epoh):
        title2.set_text(f'Epoh: {epoh + 1}')
        losers_circle = Line2D([], [], color="white", marker='o', markerfacecolor="white", markeredgecolor="#FF2800", alpha=0.3, markersize=9.5, markeredgewidth=4)
        plt.legend([losers_circle], ['Losers'], loc = 1, handletextpad=-0.1)

        for x in Xtrain:
            som.update(x, som.winner(x), epoh, epohs)

        winner_coordinates = np.array([som.winner(x) for x in Xtrain]).T
        cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

        scater1 = []

        for c in np.unique(cluster_index):
            cluster_color = plt.cm.get_cmap('tab10')(c)
            scater = ax2.scatter(Xtrain[cluster_index == c, 0], Xtrain[cluster_index == c, 1], label='cluster='+str(c), color=cluster_color)
            scater1.append(scater)

        for centroid in som.get_weights():
            scater = ax2.scatter(centroid[:, 0], centroid[:, 1], marker='o', s=8, linewidths=10, color='#FF2800', alpha=0.2)
            scater1.append(scater)

        for x, c in zip(Xtrain, cluster_index):
            i, y = som.winner(x)
            cluster_color = plt.cm.get_cmap('tab10')(c)
            winner = som.get_weights()[i, y]
            scater = ax2.scatter(winner[0], winner[1], marker='o', s=8, linewidths=10, color=cluster_color)
            scater1.append(scater)

        return *scater1, title2,

    anim = FuncAnimation(fig, update, frames=num_iteration, blit=True, interval=30, repeat=False)
    return anim




def plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

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

    winner_coordinates = np.array([som.winner(x) for x in Xtrain]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)

    ax2.set_title("Γραφική παράσταση ανταγωνιστικών νευρώνων" + (f' (Epoh {epohs})')) 
    ax2.set_xlabel("Άξονας Χ")
    ax2.set_ylabel("Άξονας Υ")
    losers_circle = Line2D([], [], color="white", marker='o', markerfacecolor="white", markeredgecolor="#FF2800", alpha=0.3, markersize=9.5, markeredgewidth=4)
    plt.legend([losers_circle], ['Losers'], loc = 1, handletextpad=-0.1)

    for c in np.unique(cluster_index):
        cluster_color = plt.cm.get_cmap('tab10')(c)
        ax2.scatter(Xtrain[cluster_index == c, 0], Xtrain[cluster_index == c, 1], label='cluster='+str(c), color=cluster_color)

    for centroid in som.get_weights():
        ax2.scatter(centroid[:, 0], centroid[:, 1], marker='o', s=8, linewidths=10, color='#FF2800', alpha=0.2)

    for x, c in zip(Xtrain, cluster_index):
        i, y = som.winner(x)
        cluster_color = plt.cm.get_cmap('tab10')(c)
        winner = som.get_weights()[i, y]
        ax2.scatter(winner[0], winner[1], marker='o', s=8, linewidths=10, color=cluster_color)

    plt.show()

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

n = 80
learning_rate = 0.5
epohs = 100
k1 = 3
k2 = 3

while True:
    option = input("\nPlease pick an option: ")
    if option.isdigit():
        option = int(option)

    if option == 1:
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData(n)

        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        k1 = int(input("Please give two numbers for the competitive neurons size, FIRST number: "))
        k2 = int(input("Please provide the SECOND number: "))

        som_shape = (k1, k2)
        som = MiniSom(som_shape[0], som_shape[1], Xtrain.shape[1], learning_rate=.5, neighborhood_function='gaussian', random_seed=10)
        #som.train_batch(Xtrain, epohs, verbose=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        anim = animation(som, Xtrain, Ltrain, Xtest, Ltest, (ax1, ax2), fig, epohs)
        plt.show()
        plot()

    elif option == 2:
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData2(n)

        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        k1 = int(input("Please give two numbers for the competitive neurons size, FIRST number: "))
        k2 = int(input("Please provide the SECOND number: "))

        som_shape = (k1, k2)
        som = MiniSom(som_shape[0], som_shape[1], Xtrain.shape[1], learning_rate=.5, neighborhood_function='gaussian', random_seed=10)
        #som.train_batch(Xtrain, epohs, verbose=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        anim = animation(som, Xtrain, Ltrain, Xtest, Ltest, (ax1, ax2), fig, epohs)
        plt.show()
        plot()

    elif option == 3:
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData3(n)

        # learning_rate = float(input("Please pick the learning rate of the algorith: "))
        # epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        k1 = int(input("Please give two numbers for the competitive neurons size, FIRST number: "))
        k2 = int(input("Please provide the SECOND number: "))

        som_shape = (k1, k2)
        som = MiniSom(som_shape[0], som_shape[1], Xtrain.shape[1], learning_rate=.5, neighborhood_function='gaussian')
        #som.train_batch(Xtrain, epohs, verbose=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        anim = animation(som, Xtrain, Ltrain, Xtest, Ltest, (ax1, ax2), fig, epohs)
        plt.show()
        plot()

    elif option == 4:
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData4(n)

        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        k1 = int(input("Please give two numbers for the competitive neurons size, FIRST number: "))
        k2 = int(input("Please provide the SECOND number: "))

        som_shape = (k1, k2)
        som = MiniSom(som_shape[0], som_shape[1], Xtrain.shape[1], learning_rate=.5, neighborhood_function='gaussian', random_seed=10)
        #som.train_batch(Xtrain, epohs, verbose=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        anim = animation(som, Xtrain, Ltrain, Xtest, Ltest, (ax1, ax2), fig, epohs)
        plt.show()
        plot()

    elif option == 5:
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData5(n)

        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        k1 = int(input("Please give two numbers for the competitive neurons size, FIRST number: "))
        k2 = int(input("Please provide the SECOND number: "))

        som_shape = (k1, k2)
        som = MiniSom(som_shape[0], som_shape[1], Xtrain.shape[1], learning_rate=.5, neighborhood_function='gaussian', random_seed=10)
        #som.train_batch(Xtrain, epohs, verbose=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        anim = animation(som, Xtrain, Ltrain, Xtest, Ltest, (ax1, ax2), fig, epohs)
        plt.show()
        plot()

    elif option == 6:
        print("Thank you for visiting, please come again.")
        break

    else:
        print("Thats not an option :(")