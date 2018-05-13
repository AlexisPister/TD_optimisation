#coding: utf8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import sqrt


def Gen_villes(L, Nvilles):
	Coord = np.random.rand(Nvilles, 2) * L
	plt.scatter(Coord[:,0], Coord[:,1], c = 'r')
	plt.savefig('Villes.png', bbox_inches='tight')
	#plt.show()
	print Coord
	return Coord


def Distance_eucl(Ville, chemin):
	#Distance euclidenne totale
	D = 0
	for i,x in enumerate(chemin[:-1]):
		d = np.sqrt(np.sum((Ville[chemin[i+1]] - Ville[x])**2))
		D +=d
	return D

#Permute 2 positions du chemin
def Permut(Chem):
	Chem_p = []
	for i in range(len(Chem)):
		Chem_p.append(Chem[i])
	P = np.random.randint(len(Chem_p), size=2)
	Chem_p[P[0]], Chem_p[P[1]] = Chem_p[P[1]], Chem_p[P[0]]
	return Chem_p



def Recuit(L, N, kappa, nb_it_max, Schema):
	#Ville = np.array([[9.92774637,15.33793624],[12.99093562,14.09356273],[4.03986254,8.84738],[10.38364537,3.87365245],[18.39374635,8.99863527],[19.39473628,12.89373645],[4.66676354,13.83673827],[3.88635627,6.09876765],[10.99986543,0.98257397],[2.99625462,17.82648274]])
	#Chemin0 = [7, 2, 8, 1, 0, 5, 4, 9, 6, 3]
	
	#Genere une ville et un chemin aléatoire
	Ville = Gen_villes(L, N)
	Chemin0 = range(N)
	np.random.shuffle(Chemin0)
	
	it=0
	cost = Distance_eucl(Ville, Chemin0)
	print "cout initial :", cost
	D_EUCL = [cost]
	
	#Stockage du plus petit chemin jamais rencontré
	Smallest = []
	for i in range(len(Chemin0)):
				Smallest.append(Chemin0[i])
				smallest_e = Distance_eucl(Ville, Chemin0)
	
	#Plot du chemin initial généré aléatoirement
	Plot_chemin = []
	for x in Chemin0:
		Plot_chemin.append(Ville[x])
	plt.scatter(Ville[:,0], Ville[:,1], c = 'r')
	plt.plot(np.array(Plot_chemin)[:,0], np.array(Plot_chemin)[:,1])
	plt.title("Chemin initial , distance parcourue = %f"%cost)
	plt.savefig('CI_%s.png'%Schema, bbox_inches='tight')
	plt.show()
	
	while it<nb_it_max:
		if Schema == "normal":
			T= 1.0/(it + 1)
		if Schema == "cube":
			T= 1.0/(it + 1)**3
		if Schema == "log":
			T= 1.0/np.log(it +1)
		print "chemin0 :", Chemin0
		Chemin1 = Permut(Chemin0)
		print "Chemin1 :", Chemin1
		
		if Distance_eucl(Ville, Chemin1)<=cost:
			Chemin0 = Chemin1
		if Distance_eucl(Ville, Chemin1)<=Distance_eucl(Ville, Smallest):
			for i in range(len(Chemin0)):
				Smallest[i] = Chemin1[i]
				smallest_e = Distance_eucl(Ville, Chemin1)
		else:
			r=np.random.random()
			if r<kappa*np.exp(-1.0/(1000.0*T)):
				Chemin0 = Chemin1
		
		cost = Distance_eucl(Ville, Chemin0)
		print "cost:", cost
		D_EUCL.append(cost)
		it+=1
		
		#Plot le nouveau chemin toute les x itérations
		#if (float(it)%5000 == 0):
		if (it == nb_it_max):
			Plot_chemin = []
			for x in Chemin0:
				Plot_chemin.append(Ville[x])
			plt.scatter(Ville[:,0], Ville[:,1], c = 'r')
			plt.plot(np.array(Plot_chemin)[:,0], np.array(Plot_chemin)[:,1])
			plt.title("n = %d , distance parcourue = %f" %(it, cost))
			plt.savefig('It_%d_%s.png'%(it,Schema), bbox_inches='tight')
			plt.show()
	plt.plot(D_EUCL)
	plt.xlabel("Iterations")
	plt.ylabel("Distance euclidienne")
	plt.savefig("Distance_e_%s.png"%Schema)
	plt.show()
	
	#Plot du plus petit chemin rencontré
	'''
	Plot_chemin = []
	for x in Smallest:
		Plot_chemin.append(Ville[x])
	plt.scatter(Ville[:,0], Ville[:,1], c = 'r')
	plt.plot(np.array(Plot_chemin)[:,0], np.array(Plot_chemin)[:,1])
	plt.title("Chemin le plus optimal rencontre, distance parcourue = %f"%smallest_e)
	plt.savefig('Opti_%s.png'%Schema, bbox_inches='tight')
	plt.show()
	'''
	return cost


L = 20
N = 10
kappa = 0.5
nb_it_max = 10000


Recuit(L, N, kappa, nb_it_max, "normal")

Recuit(L, N, kappa, nb_it_max, "cube")

Recuit(L, N, kappa, 200000, "log")



'''
Ville2 = np.random.rand(len(Ville), 2)
for i in range(len(Ville)):
	for j in range(2):
		Ville2[i][j] = Ville[i][j]

Chemin2 = []
for i in range(len(Chemin0)):
	Chemin2.append(Chemin0[i])

Ville3 = np.random.rand(len(Ville), 2)
for i in range(len(Ville)):
	for j in range(2):
		Ville3[i][j] = Ville[i][j]

Chemin3 = []
for i in range(len(Chemin0)):
	Chemin3.append(Chemin0[i])

Recuit(L, N, kappa, nb_it_max, "normal")

Ville = Ville2
Chemin0 = Chemin2

Recuit(L, N, kappa, nb_it_max, "cube")

Ville = Ville3
Chemin0 = Chemin3


Recuit(L, N, kappa, nb_it_max, "log")


Ville_Test = np.array([[1,1], [2,2], [4,4], [5,5]])
Chemin_Test = [0, 1, 2, 3]

print Distance_eucl(Ville_Test, Chemin_Test)
'''
