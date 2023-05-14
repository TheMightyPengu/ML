from matplotlib.animation import FuncAnimation
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import BigData as BD
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
import sys

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
        act = str(input("Please pick the activation function you want the algorith to use: "))
        hls = int(input("Please pick the number of hidden layers for the algorith: "))
        min_mse = float(input("Please pick the minimum MSE you want the algorith to stop at: "))
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData(n)

        clf = MLPClassifier(hidden_layer_sizes=(hls,), activation=act, solver='adam', max_iter=epohs, random_state=0, learning_rate_init=learning_rate)

    elif option == 2:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        act = str(input("Please pick the activation function you want the algorith to use: "))
        hls = int(input("Please pick the number of hidden layers for the algorith: "))
        min_mse = float(input("Please pick the minimum MSE you want the algorith to stop at: "))
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData2(n)

        clf = MLPClassifier(hidden_layer_sizes=(hls,), activation=act, solver='adam', max_iter=epohs, random_state=0, learning_rate_init=learning_rate)

    elif option == 3:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        act = str(input("Please pick the activation function you want the algorith to use: "))
        hls = int(input("Please pick the number of hidden layers for the algorith: "))
        min_mse = float(input("Please pick the minimum MSE you want the algorith to stop at: "))
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData3(n)

        clf = MLPClassifier(hidden_layer_sizes=(hls,), activation=act, solver='adam', max_iter=epohs, random_state=0, learning_rate_init=learning_rate)

    elif option == 4:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        act = str(input("Please pick the activation function you want the algorith to use: "))
        hls = int(input("Please pick the number of hidden layers for the algorith: "))
        min_mse = float(input("Please pick the minimum MSE you want the algorith to stop at: "))
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData4(n)

        clf = MLPClassifier(hidden_layer_sizes=(hls,), activation=act, solver='adam', max_iter=epohs, random_state=0, learning_rate_init=learning_rate)

    elif option == 5:
        learning_rate = float(input("Please pick the learning rate of the algorith: "))
        epohs = int(input("Please pick the number of epohs you want the algorith to run for: "))
        act = str(input("Please pick the activation function you want the algorith to use: "))
        hls = int(input("Please pick the number of hidden layers for the algorith: "))
        min_mse = float(input("Please pick the minimum MSE you want the algorith to stop at: "))
        Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData5(n)

        clf = MLPClassifier(hidden_layer_sizes=(hls,), activation=act, solver='adam', max_iter=epohs, random_state=0, learning_rate_init=learning_rate)

    elif option == 6:
        print("Thank you for visiting, please come again.")
        sys.exit()

    else:
        print("Thats not an option :(")
        
    # learning_rate = 0.005
    # epohs = 300
    # act = 'relu'
    # hls = 1000
    # min_mse = 0.05

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    def plot(epohs, clf, ax1, ax2, ax3, ax4, fig, Xtest, Ltest, Xtrain, Ltrain, min_mse, callback=None):         
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
            scatter2_1 = ax2.scatter(Xtest[predictions==1, 0], Xtest[predictions==1, 1], c='#FF00FF', marker='*', edgecolors='#FF00FF')
            scatter2_2 = ax2.scatter(Xtest[predictions==0, 0], Xtest[predictions==0, 1], marker='o', edgecolors='blue', facecolors='none')
            ax2.set_xlim([-0.015, 1.015])
            ax2.set_ylim([-0.015, 1.015])
            ax2.set_title("Διαχωριστική γραμμή")
            title2.set_text(f'Epoh: {epohs + 1}')    
            ax2.set_xlabel("Άξονας Χ")
            ax2.set_ylabel("Άξονας Υ")
            
            scatter3_1 = ax3.scatter(range(len(predictions[Ltest == 0])), predictions[Ltest == 0], marker='*', c='blue', label='Ltest == 0')
            scatter3_2 = ax3.scatter(range(len(predictions[Ltest == 0]), len(Ltest)), predictions[Ltest == 1], marker='*', c='#FF00FF', label='Ltest == 1')
            ax3.set_xlim([-0.3, Xtrain.shape[0]])
            ax3.set_ylim([-0.015, 1.015])
            ax3.set_title("Γράφημα εξόδων προτύπων")
            title3.set_text(f'Epoh: {epohs + 1}')    
            ax3.set_xlabel("Πρότυπο")
            ax3.set_ylabel("Έξοδος Y")

            line, = ax4.plot([], [])
            mse.append(np.mean((Ltrain - clf.predict(Xtrain))**2))
            line.set_data(np.arange(len(mse)), mse)
            ax4.relim()
            ax4.autoscale(True, True, True)
            ax4.set_title("MSE κατά την εκπαίδευση")
            title4.set_text(f'Epoch: {epohs + 1}\nMSE: {mse[epohs]}')
            ax4.set_xlabel("Εποχή")
            ax4.set_ylabel("MSE")
            ax4.set_xlim([0, epohs])
            ax4.relim()
            ax4.autoscale_view()
            ticks = np.arange(epohs)
            ax4.set_xticks(ticks[::3])
            ax4.set_xticklabels(ticks[::3])

            if (mse[epohs] < min_mse):
                print(f'Hooray! The model has been trained successfully. Stopped at epoh {epohs} with an MSE of {mse[epohs]}')
                cb(Xtest, Ltest, Xtrain, predictions, xx, yy, contour_predictions, epohs)
            if (epohs+1 == clf.max_iter):
                epohs = epohs + 1
                cb(Xtest, Ltest, Xtrain, predictions, xx, yy, contour_predictions, epohs)

            return scatter2_1, scatter2_2, contour2_1.collections[0], scatter3_1, scatter3_2, line, title2, title3, title4

        anim = FuncAnimation(fig, update, frames=epohs, blit=True, interval=50, repeat=False)
        return anim

    def cb(Xtest, Ltest, Xtrain, predictions, xx, yy, contour_predictions, epohs):
        plt.close()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        title2 = ax2.text(0.9, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.4, 'pad': 3}, transform=ax2.transAxes, ha="center")
        title3 = ax3.text(0.9, 0.9, "", bbox={'facecolor': 'w', 'alpha': 0.4, 'pad': 3}, transform=ax3.transAxes, ha="center")

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

        ax2.contour(xx, yy, contour_predictions, colors='black', levels=[0.5])
        ax2.scatter(Xtest[predictions==1, 0], Xtest[predictions==1, 1], c='#FF00FF', marker='*', edgecolors='#FF00FF')
        ax2.scatter(Xtest[predictions==0, 0], Xtest[predictions==0, 1], marker='o', edgecolors='blue', facecolors='none')
        ax2.set_xlim([-0.015, 1.015])
        ax2.set_ylim([-0.015, 1.015])
        ax2.set_title("Διαχωριστική γραμμή") 
        title2.set_text(f'Epoh: {epohs}') 
        ax2.set_xlabel("Άξονας Χ")
        ax2.set_ylabel("Άξονας Υ")

        ax3.scatter(range(len(predictions[Ltest == 0])), predictions[Ltest == 0], marker='*', c='blue', label='Ltest == 0')
        ax3.scatter(range(len(predictions[Ltest == 0]), len(Ltest)), predictions[Ltest == 1], marker='*', c='#FF00FF', label='Ltest == 1')
        ax3.set_xlim([-0.3, Xtrain.shape[0]])
        ax3.set_ylim([-0.015, 1.015])
        ax3.set_title("Γράφημα εξόδων προτύπων")
        title3.set_text(f'Epoh: {epohs}') 
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
        return

    callback = lambda: cb(fig, ax1, ax2, ax3, ax4, Xtest, Ltest, Xtrain, Ltrain, clf)
    anim = plot(epohs, clf, ax1, ax2, ax3, ax4, fig, Xtest, Ltest, Xtrain, Ltrain, min_mse, callback=callback)
    plt.show()