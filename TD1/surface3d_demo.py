#Imports from the matplotlib library
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import Descente
#--------------------------------------

#Definition of what to plot
fig = plt.figure() #opens a figure environment
ax = fig.gca(projection='3d') #to perform a 3D plot
X = np.arange(-5, 5, 0.25) #x range
Y = np.arange(-5, 5, 0.25) #y range
X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
#Z = (X - Y)^4 + 2*X^2 + Y^2 - X + 2*Y
#Z= X**2 - Y**2 #defines the function values
Z = X**4 - X**3 - 20*X**2 + X + 1 + Y**4 - Y**3 - 20*Y**2 + Y + 1
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options

#Trajectory
x,y = Descente.descente(-5, 5, 0.01, 1000)


#Vector of f(x,y)
z = []
for i in range(len(x)):
	#z.append((x[i] - y[i])**4 + 2*x[i]**2 + y[i]**2 - x[i] + 2*y[i])
	z.append(x[i]**2 - y[i]**2)
	

#Trace la trajectoire
#ax.plot(x, y, z, label='Trajectoire')
ax.legend() #adds a legend


#Runs the plot command
plt.show()
