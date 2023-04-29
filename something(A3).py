import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def gradient(x, y):
    s = np.sqrt(2)
    return np.array([x/np.sqrt(x**2 + y**2 + s**2), y/np.sqrt(x**2 + y**2 + s**2)])

def update_test_point(XY, learning_rate):
    grad_obj = gradient(XY[0], XY[1])
    XYnew = XY - learning_rate * grad_obj
    obj = np.sqrt(XYnew[0]**2 + XYnew[1]**2 + s**2)
    return XYnew, obj

learning_rate = float(input("Please pick the learning rate of the algorithm: "))
ranges = float(input("Enter range value a [-a,a]: "))

x = np.arange(-ranges, ranges + 0.1, 0.1, dtype=float)
y = np.arange(-ranges, ranges + 0.1, 0.1, dtype=float)
x, y = np.meshgrid(x, y)

s = np.sqrt(2)
objective = np.sqrt(x**2 + y**2 + s**2)

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(4, -4)
ax.set_ylim(4, -4)
ax.view_init(elev=23, azim=43)

color_threshold = [0.0, 0.25, 0.5, 0.75, 1.0]
colors = ['blue', 'teal', 'yellow', 'orange', 'red']
custom = mcolors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(color_threshold, colors)))
surf = ax.plot_surface(x, y, objective, cmap=custom, alpha=0.6)

XYtest = np.array([ranges, ranges])
ObjTest = np.sqrt(XYtest[0]**2 + XYtest[1]**2 + s**2)
dot = ax.scatter(XYtest[0], XYtest[1], ObjTest, color='black', s=150, marker='o', alpha=1)

satisfied = 0.000001
old_objTest = ObjTest + 1
while abs(ObjTest - old_objTest) > satisfied:
    old_objTest = ObjTest
    XYtest, ObjTest = update_test_point(XYtest, learning_rate)
    dot._offsets3d = ([XYtest[0]], [XYtest[1]], [ObjTest])
    ax.legend([dot], [f'Coordinates:  (x:{XYtest[0]:.2f}, y:{XYtest[1]:.2f})'])
    plt.draw()
    plt.pause(0.1)

ax.legend([dot], [f'Ended at:  (x:{XYtest[0]:.2f}, y:{XYtest[1]:.2f})'])
plt.show(block=True)
