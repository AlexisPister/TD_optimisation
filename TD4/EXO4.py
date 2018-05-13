#coding: utf8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#1.1
def f(x):
	return x**4-x**3-20*x**2+x+1

#2.1
def g(x,y):
	return f(x)+f(y)


def recuit_simule(nb_it_max,kappa,kappaprime,x0,y0,m):
	X = [x0]
	Y = [y0]
	it=0
	cost=g(x0,y0)
	COST = [cost]
	while it<nb_it_max:
		T=1/(it+1.0)
		couts_simul = []
		for i in range(m):
			Dprime=np.random.normal(0,kappa*np.exp(-1.0/(1000*T)),size=2)
			xprime = x0 + Dprime[0]
			yprime = y0 + Dprime[1]
			couts_simul.append(g(xprime, yprime))
		Delta = 1.0/m * np.sum(np.array(couts_simul) - cost)
		
		
		D=np.random.normal(0,kappa*np.exp(-1.0/(1000*T)),size=2)
		x1=x0+D[0]
		y1=y0+D[1]

		if g(x1,y1)<=cost:
			x0=x1
			y0=y1
			cost=g(x0,y0)
		else:
			r=np.random.random()
			if r<kappaprime*np.exp(-Delta/(1000.0*T)):
				x0 = x1
				y0 = y1
				cost = g(x0, y0)
		X.append(x0)
		Y.append(y0)
		COST.append(cost)
		it+=1
	
	plt.plot(X)
	plt.plot(Y)
	plt.show()
	#return np.array(X), np.array(Y)
	return x0, y0

nb_it_max=10000
kappa=10
kappaprime=0.5
x0=0
y0=0
m=5

recuit_simule(nb_it_max,kappa,kappaprime,x0,y0,m)
'''
tentatives=50
xy_finaux=np.zeros((tentatives,2))
for i in range(tentatives):
	result=recuit_simule(nb_it_max,kappa,kappaprime,x0,y0, m)
	xy_finaux[i,:]=result
plt.hist2d(xy_finaux[:,0],xy_finaux[:,1])
#plt.axis([3.525, 3.575, 3.525, 3.575])
plt.colorbar()
plt.show()
'''


#PLOT de la fonction 2d et de la trajectoire
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
x=np.linspace(-5,5,5*10+1)
y=np.linspace(-5,5,5*10+1)
X, Y = np.meshgrid(x,y) #creates a grid
Z= g(X,Y) #defines the function values
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
					   linewidth=0, antialiased=False)

STEPS = recuit_simule(nb_it_max,kappa,kappaprime,x0,y0)
xn=STEPS[0]
yn=STEPS[1]
zn=g(xn,yn)
ax.plot(xn, yn, zn,'-+',color="white") #

plt.show()
'''
