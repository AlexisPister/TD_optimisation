import numpy as np
import matplotlib.pyplot as plt
import random


#Generateur de villes
def villes(N):
	liste_coord = []
	L = 10
	for ville in range(N):
		liste_coord.append(L*np.random.rand(2,1))
	return liste_coord


def f(trajet,coord):
	list_tri = [coord[i] for i in trajet]
	dist1 = 0
	for i in range(len(trajet)-1):
		dist1 = dist1 + np.sqrt((list_tri[i+1][0]-list_tri[i][0])**2+(list_tri[i+1][1]-list_tri[i][1])**2)
        """
	dist = 0
	for i in range(len(trajet)-1):
		dist = dist + np.sqrt((coord[trajet[i+1]][0]-coord[trajet[i]][0])**2+(coord[trajet[i+1]][1]-coord[trajet[i]][1])**2)
	"""
	return dist1


def permutation(t):
	t_bis = []
	for i in range(len(t)):
		t_bis.append(t[i])
	perm = np.random.randint(len(t),size=2)
	val1 = t_bis[perm[0]]
	val2 = t_bis[perm[1]]
	t_bis[perm[1]] = val1
	t_bis[perm[0]] = val2
	return t_bis

def coor_tri(trajet,coord):
	list_tri = [coord[i] for i in trajet]
	return list_tri

coord = villes(20) #coordonnees des villes


fig = plt.figure()
ax = fig.gca()
"""

x = np.linspace(-6,6,1201)
#ax.plot(x,f(x))
#plt.show()
"""




def alg_recuit(tmax, k, k_b) : 
	traj = np.arange(20)
	print traj
	trajet_list = []
	f_list = []
	for t in range(2,tmax):	
		#if t in range(1,tmax,100):
		#	print 'distance : ', f(traj,coord)
		#	coord_t = coor_tri(traj,coord)
		#	for i in range(len(coord_t)-1):
		#		plt.plot([coord_t[i][0],coord_t[i+1][0] ], [coord_t[i][1],coord_t[i+1][1] ], 'r-', lw=2)
		#	x = [coord_t[i][0] for i in range(len(coord_t))]
		#	y = [coord_t[i][1] for i in range(len(coord_t))]
		#	plt.scatter(x,y,s=100)
		#	plt.show()
		
		T = 1.0/np.log(t)
		t_ev = permutation(traj)
		fdiff_list = []
		m = 5
		#for i in range(m):
		#	trajet_i = permutation(traj)
		#	fdiff = f(trajet_i,coord) - f(traj,coord)
		#	fdiff_list.append(fdiff)
		#E = (1.0/m)*np.sum(fdiff_list)
		#print '###########################'
		#print traj
		#print t_ev
		#print f(t_ev,coord)
		#print f(traj,coord)
		if f(t_ev,coord) < f(traj,coord) :
			traj = t_ev
		else :
			if random.uniform(0,1) < k_b*np.exp(-1/(1000*T)):
				traj = t_ev
		trajet_list.append(traj)
		f_list.append(f(traj,coord))
		print f(traj,coord)
		print t
		if (len(f_list) > 10) and (np.var(f_list[-20:]) == 0.0) :
			print f_list
			print t
			break
		#print trajet, coord
		#print k_b*np.exp(-1/(1000*T))

	coord_t = coor_tri(traj,coord)
	for i in range(len(coord_t)-1):
		plt.plot([coord_t[i][0],coord_t[i+1][0] ], [coord_t[i][1],coord_t[i+1][1] ], 'r-', lw=2)
	x = [coord_t[i][0] for i in range(len(coord_t))]
	y = [coord_t[i][1] for i in range(len(coord_t))]
	plt.scatter(x,y,s=100)
	plt.show()

	#ax.plot(f_list)
	#plt.show()
	return traj


trajet_t = alg_recuit(20000,10,0.5)
print trajet_t
"""
x_term = []

for tmax in range(6000,12000,1000) :
	for k in range(5,20,5) :
		for k_b in np.arange(1,10,1)/10.0 :
			trajet_t = alg_recuit(tmax,k,k_b)
			trajet_term.append(xterm)
			print xterm , tmax , k , k_b

hist, bins = np.histogram(x_term, bins=10)
width = 0.3 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
"""
#Variations des valeurs de x, au debut les valeurs de D varient beaucoup (T grand) mais quand elles diminuent on atteind des variations moindres.











	

		
