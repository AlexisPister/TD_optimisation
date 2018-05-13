# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def g(x, a):
	return np.exp(-a*x)
	
def f(x, y, a):
	return 0.5 * np.sum((y-g(x,a))**2)
	
#dg/da
def grad_g(x, a):
	return -x * np.exp(-a*x)

#df/da
def grad_f(x, y, a):
	return -np.sum((y - g(x,a)) * grad_g(x, a))
	
def hess_f(x, y, a):
	return np.sum(grad_g(x,a)**2)
	
a = 2.0
b = 0.1


#Creation du jeu de donné
x = np.linspace(0, 3, 301)
plt.plot(x, g(x, a))

y = g(x,a) + b * np.random.randn(301)
plt.plot(x, y, '+')
#plt.show()


#Algorithme de Levenberg-Marquardt
a = 1.5
lbd = 0.001
fk = f(x, y, a)
F = [fk]
n = 0
c = 0
while n < 200:
	print fk
	grad = grad_f(x, y, a)
	hess = hess_f(x, y, a)
	#Nouvelle hessienne
	while c == 1:
		c = 0
		H = hess * (1 + lbd)
		d = -grad/H
	
		ftest = fk + d
		print ftest
		F.append(Ftest)
		#On test si le nouveau f est plus petit que l'ancien
		if ftest < fk:
			a = a + d
			lbd = lbd/10
			c = 0
		elif ftest > fk:
			lbd = lbd*10
			c = 1
		fk = ftest
		n += 1
		
print "a :", a, "lambda :", lbd
		
def levenberg(a, x, y, lam, nb_iter_max, norm_min):
	nb_iter = 0
	a0 = a
	f0 = f(x,y,a)
		
		
		
		
		
		
