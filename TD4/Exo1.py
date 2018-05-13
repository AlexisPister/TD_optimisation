#TD1 > 14/02
#TD2 > 28/02
#TD4 ex3 et 4 > 07/03

#coding : utf8
#Heuristique
import numpy as np
import matplotlib.pyplot as plt


def f(x):
	return x**4 - x**3 - 20*x**2 + x + 1


x = np.linspace(-6, 6, 1201)
#plt.plot(x, f(x))
#plt.show()


def recuit(f, x0, k, k2, tmax):
	t = 1
	X = [x0]
	T0 = 1.0/t
	T = [T0]
	print x0
	print T0
	while(t <= tmax):
		t+=1
		xk = x0 + float(np.random.normal(0,k*np.exp(-1.0/(1000.0*T0)),1))
		
		#Si le solution est moins bonne on la garde avec une proba
		if (f(xk) > f(x0)):
			test = np.random.random()
			if (test > k2 * np.exp(-1/(1000*T0))):
				xk = x0
				
		T0 = 1.0/t
		
		print "x : ", xk
		print "T0 : ", T0
		T.append(T0)
		X.append(xk)
		
		x0 = xk
		
	X = np.array(X)
	plt.plot(X)
	plt.show()
	return X[tmax]


recuit(f, -1.0, 10, 0.5, 10000)

'''
tentatives = 5
x_finaux = np.zeros((tentatives, 1))
for i in range(tentatives):
	x_finaux[i] = recuit(f, -1.0, 10, 0.5, 10000)
	
plt.hist(x_finaux)
plt.show() 	
'''
		
		

		
