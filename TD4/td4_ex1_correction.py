#coding : utf8
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#fonction f
def f(x):
    return x**4-x**3-20*x**2+x+1

#pour plotter la fonction
x=np.linspace(-6,6,1201)
plt.plot(x,f(x))
plt.show()

steps = []
#fonction du recuit
def recuit_simule(nb_it_max,kappa,kappaprime,x0):
    t=0
    cost=f(x0)

    #condition d'arret : nombre d'iterations
    while t<nb_it_max:
        T=1/(t+1.0) #maj de la temperature
        print 'T:',T
        D=np.random.normal(0,kappa*np.exp(-1.0/(1000*T)),size=1) #calcul du deplacement
        x1=x0+D #deplacement de x

        if f(x1)<=cost: #si on a un cout inferieur, on garde
            x0=x1
            cost=f(x1)
        else: #sinon, on garde avec une certaine proba
            r=np.random.random()
            if r<kappaprime*np.exp(-1.0/(1000*T)):
                x0 = x1
                cost = f(x1)
        steps.append(x0)
        t+=1
    return x0

#parametres
nb_it_max=10000
kappa=10.0
kappaprime=0.5
x0=-1

print recuit_simule(nb_it_max,kappa,kappaprime,x0)


'''
#lancer plusieurs fois l'algorithme pour voir la tendance de ses convergences
#recuperer la position finale a chaque fois
tentatives=50
x_finaux=np.zeros((tentatives,1))
for i in range(tentatives):
    steps=[]
    [x_final]=recuit_simule(nb_it_max,kappa,kappaprime,x0)
    x_finaux[i,:]=[x_final]
plt.hist(x_finaux) #plotter l'histogramme
plt.show()

#plotter la trajectoire du dernier run de l'aglorithme que l'on a fait
steps=np.array(steps)
plt.plot(steps)
plt.show()
'''
