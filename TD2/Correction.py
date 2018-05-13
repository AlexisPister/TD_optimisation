#coding : utf8

import numpy as np
import matplotlib.pyplot as plt


#Q1 :
#avantages :
#facile a implementer
#economes en calcul
#progressent a chaque iteration vers un minimum

#inconvenients :
#restent bloquees dans les minima locaux
#descente du gradient peut etre tres long a converger, notamment dans le cas de "plaines"
#descente du gradient necessite pas a fixer carefully
#methode de newton ne fait pas de distinction entre minima, maxima, point selle (mise a zero de l'approximation quadratique)

#Q2
def g(x,a):
    return np.exp(-a*x)

#Q3
a_true=2
b=0.1
x=np.linspace(0,3,301)
plt.plot(x,g(x,a_true))
# plt.show()
y=g(x,a_true)+b*np.random.randn(301)
plt.plot(x,y,'+')
plt.show()

#Q5 :
def f(x,y,a):
    return (1/2)*np.sum((y-g(x,a))**2)

#Q6 : attention, on derive bien par rapport a a
# Pour l'instant c'est facile car a est de dimension 1
def grad_g(x,a):
    return -x*np.exp(-a*x)

def grad_f(x,y,a):
    return -np.sum((y-g(x,a))*grad_g(x,a))

#Q7
def hess_f(x,a):
    return np.sum(grad_g(x,a)**2)

#Q8
def norm_grad(a):
    return np.sqrt(a**2)

#vecteurs pour suivre l'evolution des valeurs de l'algorithme au fil des iterations
error=[]
lambdas=[]
norms=[]
aa=[]

def levenberg(a,x,y,lam,nb_iter_max,norm_min):
    nb_iter=0
    a0=a
    f0=f(x,y,a)
    dlm=norm_min

    #conditons d'arret
    while(nb_iter<nb_iter_max and norm_grad(dlm)>=norm_min):
        gradf=grad_f(x,y,a0)
        hessf=hess_f(x,a0)*(1+lam)
        dlm=-gradf/hessf

        #maj de a
        a1=a0+dlm
        #calcul du nouveau cout
        f1=f(x,y,a1)

        #si on descend : on continue en Newton (baisse de lambda)
        if f1<f0:
            f0=f1
            a0=a1
            lam=lam/10
        #si on remonte : pas bon, on ne garde pas cette iteration et on tend + vers la descente du gradient (augmentation de lambda)
        else:
            lam=lam*10

        #garde des valeurs pour etudier leur evolution au fil des iterations
        error.append(f0)
        lambdas.append(lam)
        norms.append(norm_grad(gradf))
        aa.append(a0)

        nb_iter+=1

    #apres convergence
    print(" ")
    print(">>>>>>")
    print("Levenberg descent ended with following results")
    print("a is "+str(a0))
    print("f(a) is "+str(f0))
    print("nb_iteration is "+str(nb_iter)+" while nb_iter_max was "+str(nb_iter_max))
    print("gradient norm is "+str(norm_grad(dlm))+" while gradient norm min was "+str(norm_min))
    return a0


nb_iter_max=200
norm_min=1e-5
lam=0.001
a=1.5

a_final=levenberg(a,x,y,lam,nb_iter_max,norm_min)
# plt.plot(np.log10(lambdas))
plt.plot(error)
# plt.plot(norms)
# plt.show()

