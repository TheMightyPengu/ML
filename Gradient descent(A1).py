import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def f(x):
    fx = (x**3) - (3*x**2)
    return fx

def df(x):
    dfx = (3*x**2) - (6*x)
    return dfx

def update(frame, learning_rate, point, start):
    current_point = point.get_data()[0]
    gradient = df(current_point)

    new_point = current_point - learning_rate * gradient
    point.set_data(new_point, f(new_point))
    return [point]

x = np.arange(-1, 4.1, 0.1)

fig, ax = plt.subplots()

fx = f(x)
dfx = df(x)

while True:
    start = input("Give me a positive number as a starting point: ")
    if start.isdigit():
        start = float(start)
        if start > 0:
            break
        else:
            print("Please enter a positive number.")
    else:
        print("That's not what I asked for. Please enter a positive number.")

plt.plot(x, dfx, label='fp(x)', color='red')
plt.plot(x, fx, label='f(x)', color='blue')
point = plt.plot(start, f(start), "ro", markersize=10)[0]

plt.xlabel("x")
plt.ylabel("f(x) fp(x)")
plt.legend()
plt.title("Gradient Descent f(x)=x^3-3x^2")

plt.xlim([min(x), max(x)])
ax.set_xticks(np.arange(-1, 4.1, 0.5))

ax.axhline(0, color='gray', linewidth=0.45)
ax.axvline(0, color='gray', linewidth=0.45)

learning_rate = 0.02
animations = animation.FuncAnimation(fig, update, frames=300, interval=300, blit=True, fargs=(learning_rate, point, start))
plt.show()
