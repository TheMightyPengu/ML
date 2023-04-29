import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

learning_rate = float(input("Please pick the learning rate of the algorith: "))
ranges = float(input("Enter range value a [-a,a]: "))

x = np.arange(-ranges, ranges + 0.1, 0.1, dtype=float)
y = np.arange(-ranges, ranges + 0.1, 0.1, dtype=float)
x, y = np.meshgrid(x, y)
objective = np.exp(-(x ** 2 + y ** 2) / 2)

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(4, -4)
ax.set_ylim(4, -4)
ax.view_init(elev=40, azim=35)

color_threshold = [0.0, 0.25, 0.5, 0.75, 1.0]
colors = ['blue', 'teal', 'yellow', 'orange', 'red']
custom = mcolors.LinearSegmentedColormap.from_list('custom_cmap', list(zip(color_threshold, colors)))
surf = ax.plot_surface(x, y, objective, cmap=custom, alpha=0.6)

XYtest = np.array([-0.1, -0.1])
ObjTest = np.exp(-(XYtest[0] ** 2 + XYtest[1] ** 2) / 2)
dot = ax.scatter(XYtest[0], XYtest[1], ObjTest, color='black', s=150, marker='o', alpha=1)

def update_point(XY, learning_rate):
    obj = np.exp(-(XY[0] ** 2 + XY[1] ** 2) / 2)
    gradObj = np.array([-XY[k] * np.exp(-(XY[0] ** 2 + XY[1] ** 2) / 2) for k in range(2)])
    XYnew = XY - learning_rate * gradObj
    return XYnew, obj


while abs(ObjTest) > 0.05:
    XYtest, ObjTest = update_point(XYtest, learning_rate)
    dot._offsets3d = ([XYtest[0]], [XYtest[1]], [ObjTest])
    ax.legend([dot], [f'Coordinates:  (x:{XYtest[0]:.2f}, y:{XYtest[1]:.2f})'])
    plt.draw()
    plt.pause(0.1)

ax.legend([dot], [f'Ended at:  (x:{XYtest[0]:.2f}, y:{XYtest[1]:.2f})'])
plt.show(block=True)