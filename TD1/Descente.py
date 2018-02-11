# coding: utf-8
import numpy as np
from decimal import *
from math import *


#Techniques de descente de gradient d'ordre 1 et 2 pour 3 fonctions différentes 

def f(x,y):
 return (x - y)^4 + 2*x^2 + y^2 - x + 2*y

#Norme du gradient en 2D
def norme(vecteur):
	return sqrt(vecteur[0]**2 + vecteur[1]**2)
	

#Methode de la plus forte pente : d = -gradient
#2D
def descente(x0, y0, pas, n):
	X = [x0]
	Y = [y0]
	xk = x0
	yk = y0
	i = 0
	#La boucle tourne n fois (methode nb d'iteration)
	while i<n:
	
	#grad = [4*(xk - yk)**3 + 4*xk - 1, 4*(yk - xk)**3 + 2*yk + 2]
	#Critere sur la norme du gradient
	#while (norme(grad) >= 4.4408920986*10**-16):
		
		#grad = [4*(xk - yk)**3 + 4*xk - 1, 4*(yk - xk)**3 + 2*yk + 2]
		#grad = [2*xk, -2*yk]
		grad = [4*xk**3 - 3*xk**2 - 40*xk +1, 4*yk**3 - 3*yk**2 - 40*yk + 1]
		d = [-x for x in grad]
		
		print (xk, yk)
		print "gradient : ", norme(grad)
		#On calcule les nouvelles valeurs
		xk = xk + pas * d[0]
		yk = yk + pas * d[1]
		X.append(xk)
		Y.append(yk)
		i += 1
		
	return (X, Y)
	
	
#descente(1, 1, 0.09, 130)
#descente(-5, 5, 0.01, 1500)
#descente(-4, -3, 0.01, 30)


#Methode de descente de gradient de second ordre
def Newton(x0, y0, n):
	X = [x0]
	Y = [y0]
	
	xk = x0
	yk = y0
	
	#Methode nb d'iterations
	i = 0
	while i<n:
		#grad = np.array([4*(xk - yk)**3 + 4*xk - 1, 4*(yk - xk)**3 + 2*yk + 2])
		#grad = np.array([2*xk, -2*yk])
		grad = np.array([4*xk**3 - 3*xk**2 - 40*xk +1, 4*yk**3 - 3*yk**2 - 40*yk + 1])
		
		print (xk, yk)
		print "gradient : ", norme(grad)
		#Calcul de l'inverse de la hessienne
		#hess = np.array([12*(xk - yk)**2 + 4, -12*(xk - yk)**2, -12*(yk - xk)**2, 12*(yk-xk)**2 + 2])
		#hess = np.array([2, 0, 0, -2])
		hess = np.array([12*xk**2 - 6*xk - 40, 0, 0, 12*yk**2 - 6*yk -40])
		hess = np.reshape(hess, (2,2))
		hessI = np.linalg.inv(hess)
		
		#Calucl de d, la distance qu'on va parcourir
		d = np.dot(grad, hessI) * -1
		
		#Descente de la fonction
		xk = xk + d[0]
		yk = yk + d[1]
		X.append(xk)
		Y.append(yk)
		
		i+=1
		
	return(X,Y)
		
		
#Newton(1, 1, 30)
#Newton(-5,5, 100)
Newton(-4, -3, 30)
