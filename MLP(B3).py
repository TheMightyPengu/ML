from matplotlib.animation import FuncAnimation
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import BigData as BD
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.colors import ListedColormap
import time

class MLP:
    n = 80
    Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData3(n)

    clf = MLPClassifier(hidden_layer_sizes=(1000,), activation='relu', solver='adam', max_iter=1000, random_state=0, learning_rate_init=0.01)
    epohs = clf.max_iter

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    def plot(epohs, clf, ax1, ax2, ax3, ax4, fig, Xtest, Ltest, Xtrain, Ltrain, n, callback=None):         
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

        mse = []
        title2 = ax2.text(0.9, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.4, 'pad': 3}, transform=ax2.transAxes, ha="center")
        title3 = ax3.text(0.9, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.4, 'pad': 3}, transform=ax3.transAxes, ha="center")
        title4 = ax4.text(0.9, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.4, 'pad': 3}, transform=ax4.transAxes, ha="center")
        def update(epohs):
            clf.partial_fit(Xtrain, Ltrain, classes=[0, 1])
            predictions = clf.predict(Xtest)
            
            Xmin, Xmax = Xtest[:, 0].min() - 0.1, Xtest[:, 0].max() + 0.1
            Ymin, Ymax = Xtest[:, 1].min() - 0.1, Xtest[:, 1].max() + 0.1
            xx = np.linspace(Xmin, Xmax, 100)
            yy = np.linspace(Ymin, Ymax, 100)
            xx, yy = np.meshgrid(xx, yy)
            grid = np.column_stack((xx.ravel(), yy.ravel()))

            contour_predictions = clf.predict_proba(grid)[:, 1]
            contour_predictions = contour_predictions.reshape(xx.shape)            
            contour2_1 = ax2.contour(xx, yy, contour_predictions, colors='black', levels=[0.5])

            label_colors = ['blue' if label == 0 else '#FF00FF' for label in predictions]
            scatter2_1 = ax2.scatter(Xtest[:,0], Xtest[:,1], c=label_colors, marker = '*', facecolors='none')
            ax2.set_xlim([-0.015, 1.015])
            ax2.set_ylim([-0.015, 1.015])
            ax2.set_title("Διαχωριστική γραμμή")
            title2.set_text(f'Epoch {epohs + 1}')    
            ax2.set_xlabel("Άξονας Χ")
            ax2.set_ylabel("Άξονας Υ")
            
            scatter3_1 = ax3.scatter(range(len(predictions[Ltest == 0])), predictions[Ltest == 0], marker='*', c='blue', label='y_test == 0')
            scatter3_2 = ax3.scatter(range(len(predictions[Ltest == 0]), len(Ltest)), predictions[Ltest == 1], marker='*', c='#FF00FF', label='Ltest == 1')
            ax3.set_xlim([-0.3, Xtrain.shape[0]])
            ax3.set_ylim([-0.015, 1.015])
            ax3.set_title("Γράφημα εξόδων προτύπων")
            title3.set_text(f'Epoch {epohs + 1}')    
            ax3.set_xlabel("Πρότυπο")
            ax3.set_ylabel("Έξοδος Y")

            line, = ax4.plot([], [])
            mse.append(np.mean((Ltrain - clf.predict(Xtrain))**2))
            line.set_data(np.arange(len(mse)), mse)
            ax4.relim()
            ax4.autoscale(True, True, True)
            ax4.set_title("MSE κατά την εκπαίδευση")
            title4.set_text(f'Epoch {epohs + 1}')    
            ax4.set_xlabel("Εποχή")
            ax4.set_ylabel("MSE")
            ax4.set_xlim([0, epohs])
            ax4.relim()
            ax4.autoscale_view()
            ticks = np.arange(epohs)
            ax4.set_xticks(ticks[::3])
            ax4.set_xticklabels(ticks[::3])

            return scatter2_1, contour2_1.collections[0], scatter3_1, scatter3_2, line, title2, title3, title4
        
        anim = FuncAnimation(fig, update, frames=epohs, blit=True, interval=50, repeat=False)
        return anim
    
    anim = plot(epohs, clf, ax1, ax2, ax3, ax4, fig, Xtest, Ltest, Xtrain, Ltrain, n)
    plt.show()



####################




# while True:
#     n = input("Please give a number that's a multiple of 8: ")
#     if n.isdigit() and int(n) % 8 == 0:
#         n = int(n)
#         break

# print('''
# Your options are:
# 1. Γραμμικά Διαχωρίσιμα Πρότυπα
# 2. Μη Γραμμικά Διαχωρίσιμα Πρότυπα – Κλάση 0 στη Γωνία
# 3. Μη Γραμμικά Διαχωρίσιμα Πρότυπα - Κλάση 0 στο Κέντρο
# 4. Μη Γραμμικά Διαχωρίσιμα Πρότυπα - Πύλη XOR
# 5. Μη Γραμμικά Διαχωρίσιμα Πρότυπα - Κλάση 0 μέσα στην Κλάση 1
# 6. Τέλος
# ''')

# while True:
#     option = input("\nPlease pick an option: ")
#     if option.isdigit():
#         option = int(option)

#     if option == 1:
#         learning_rate = float(input("Please pick the learning rate of the algorith: "))
#         epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

#         Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData(n)
#         mlp = MLP(learning_rate, epohs)


#     elif option == 2:
#         learning_rate = float(input("Please pick the learning rate of the algorith: "))
#         epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

#         Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData2(n)
#         mlp = MLP(learning_rate, epohs)


#     elif option == 3:
#         learning_rate = float(input("Please pick the learning rate of the algorith: "))
#         epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

#         Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData3(n)
#         mlp = MLP(learning_rate, epohs)


#     elif option == 4:
#         learning_rate = float(input("Please pick the learning rate of the algorith: "))
#         epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

#         Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData4(n)
#         mlp = MLP(learning_rate, epohs)

#     elif option == 5:

#         learning_rate = float(input("Please pick the learning rate of the algorith: "))
#         epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))

#         Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData5(n)
#         mlp = MLP(learning_rate, epohs)

#     elif option == 6:
#         print("Thank you for visiting, please come again.")
#         break

#     else:
#         print("Thats not an option :(")