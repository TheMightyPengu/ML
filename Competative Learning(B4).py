import numpy as np
import lines as l
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import BigData as BD
import sys


class Competitive_Learning:
    def __init__(self, nebrones, lr):
        self.nebrones = nebrones
        self.w=np.random.random_sample((self.nebrones, x.shape[1]))
        self.lr=lr

    def partial_fit(self, x, epohs, current_epoh):
        norms = np.linalg.norm(self.w - np.expand_dims(x,0), axis = -1) 
        indx = norms.argmin()
        self.w[indx] += self.lr * (x-self.w[indx])
        self.lr *= (1-(current_epoh/epohs))
        
    def plot(self, Xtrain, Xtest, Ltest, epohs, training=False):
        fig, ax1 = plt.subplots()

        for i in range(epohs):
            for j,x in enumerate(Xtrain):
                if training:
                    self.partial_fit(x, epohs, i)

            ax1.scatter(self.w[:, 0], self.w[:, 1], marker='o', color='black', edgecolors='black', s=100)

        for i in range(len(Xtest)):
            if Ltest[i] == 0:
                ax1.scatter(Xtest[i][0], Xtest[i][1], marker='o', color='blue', edgecolors='blue', facecolors='none')
            else:
                ax1.scatter(Xtest[i][0], Xtest[i][1], marker='*', color='#FF00FF')

        ax1.set_xlim([-0.015, 1.015])
        ax1.set_ylim([-0.015, 1.015])
        ax1.set_title("Διαχωριστική γραμμή")
        ax1.set_xlabel("Άξονας Χ")
        ax1.set_ylabel("Άξονας Υ")

        plt.plot()


     


choice=int(input('1->Grammika Diaxorisima\n2->Goneia\n3->XOR\n4->Kentro\n5->Grammika Diaxorisima 3d \
\n6->Xor \n'))
x,y=None,None

if choice==1:Xtrain,Xtest,Ltrain,Ltest=BD.create_BigData(int(input('Arithmos protipon:')))
elif choice==2:Xtrain,Xtest,Ltrain,Ltest=BD.create_BigData2(int(input('Arithmos protipon:')))
elif choice==3:Xtrain,Xtest,Ltrain,Ltest=BD.create_BigData3(int(input('Arithmos protipon:')))
elif choice==4:Xtrain,Xtest,Ltrain,Ltest=BD.create_BigData4(int(input('Arithmos protipon:')))
elif choice==5:Xtrain,Xtest,Ltrain,Ltest=BD.create_BigData5(int(input('Arithmos protipon:')))
else: raise ValueError('Invalid Input')

x = Xtrain
y = Ltrain

lr = float(input('Dwse arxiko vima ekpedeusis: '))
nebrones= int(input('Dwse arithmo antagonistikon neuronon : '))
model = Competitive_Learning(nebrones, lr)
epohs = int(input('dose epoxes: '))
model.plot(Xtrain, Xtest, Ltest, epohs, training = True)





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
#         Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData(n)

#     elif option == 2:
#         Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData2(n)

#     elif option == 3:
#         Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData3(n)

#     elif option == 4:
#         Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData4(n)

#     elif option == 5:
#         Xtrain, Xtest, Ltrain, Ltest = BD.create_BigData5(n)

#     elif option == 6:
#         print("Thank you for visiting, please come again.")
#         sys.exit()