import numpy as np
import matplotlib.pyplot as plt
import random

def f(x):
	return x**4-x**3-20*x**2+x+1

fig = plt.figure()
ax = fig.gca()
x = np.linspace(-6,6,1201)
#ax.plot(x,f(x))
#plt.show()

#Le minimum global se trouve en x = 3.7 et on voit egalement un minimum local 
#en -3.5.

#Algorithme de recherche du minimum

def alg_recuit(tmax, k, k_b) : 
	x = -1
	x_list = []

	for t in range(1,tmax):
		T = 1.0/t
		mu = 0
		sigma = k*np.exp(-1/(1000*T))
		print "########################"
		D = np.random.randn(1) * sigma + mu
		print D
		x_ev = x + D

		fdiff_list = []
		m = 5
		for i in range(m):
			D_test = np.random.randn(1) * sigma + mu
			xi = x + D_test
			fdiff = f(xi) - f(x)
			fdiff_list.append(fdiff)
		E = (1.0/m)*np.sum(fdiff_list)
		#print 'E = ',E
		#print 'T = ',T
		#print 'P = ',k_b*np.exp(-E/(1000*T))
		#print "#####################"
		if f(x_ev) < f(x) :
			x = x_ev
			print 'sit 1'
		else :
			print 'sit 2'
			if random.uniform(0,1) < k_b*np.exp(-E/(1000*T)):
				print '#######################################################################################'
				x = x_ev
		x_list.append(x)
		print x

	ax.plot(x_list)
	plt.show()
	return x_list[-1]


alg_recuit(10000,13,0.7)
"""

x_term = []

for tmax in range(6000,12000,1000) :
	for k in range(5,20,5) :
		for k_b in np.arange(1,10,1)/10.0 :
			xterm = alg_recuit(tmax,k,k_b)
			x_term.append(xterm)
			print xterm , tmax , k , k_b

hist, bins = np.histogram(x_term, bins=10)
width = 0.3 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()

#Variations des valeurs de x, au debut les valeurs de D varient beaucoup (T grand) mais quand elles diminuent on atteind des variations moindres.

"""









	


