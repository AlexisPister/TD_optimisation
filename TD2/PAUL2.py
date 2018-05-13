#coding : utf8

import numpy as np
import matplotlib.pyplot as plt
import pylab
from numpy.linalg import inv

###Q1 :
#Fonction g
def g(x,a1,a2):
    return x**a1*np.exp(-a2*x)

#Jdd bruite
a_1=1.5
a_2=1.5
b=0.1
x=np.linspace(0.01,5,501)
pylab.plot(x,g(x,a_1,a_2),label='g(xi,a)')
y=g(x,a_1,a_2)+b*np.random.randn(501)
pylab.plot(x,y,'+',label='yi')
pylab.legend(loc='upper left')
pylab.show()

#Fonction f
def f(x,y,a1,a2):
    return 0.5*np.sum((y-g(x,a1,a2))**2)

###Q2

def grad_g(x,a1,a2):
    return [np.exp(-a2*x)*np.log(x)*x**(a1), -x**(a1)*x*np.exp(-a2*x)]

def grad_f(x,y,a1,a2):
    f_gr = []
    f_gr.append(-np.sum((y-g(x,a1,a2))*grad_g(x,a1,a2)[0]))
    f_gr.append(-np.sum((y-g(x,a1,a2))*grad_g(x,a1,a2)[1]))
    return f_gr



###Q3
def hess_f(x,a1, a2):
    h11 = np.dot(grad_g(x,a1,a2)[0],grad_g(x,a1,a2)[0])
    h12 = np.dot(grad_g(x,a1,a2)[0],grad_g(x,a1,a2)[1])
    h21 = np.dot(grad_g(x,a1,a2)[1],grad_g(x,a1,a2)[0])
    h22 = np.dot(grad_g(x,a1,a2)[1],grad_g(x,a1,a2)[1])
    return  [[h11,h12],[h21,h22]]
    


###Q4
def norm_grad(a):
    return np.sqrt(np.sum(np.dot(a,a)))

#vecteurs pour suivre l'evolution des valeurs de l'algorithme au fil des iterations
error=[]
lambdas=[]
norms=[]
aa=[]

def levenberg(a,x,y,lam,nb_iter_max,norm_min):
    nb_iter=0
    a0=a
    f0=f(x,y,a[0],a[1])
    dlm=norm_min

    #conditons d'arret
    while(nb_iter<nb_iter_max and norm_grad(dlm)>=norm_min):
        gradf=grad_f(x,y,a0[0],a0[1])
        hessf=hess_f(x,a0[0],a0[1])
	hessf[0][0] = hessf[0][0]*(1+lam)
	hessf[1][1] = hessf[1][1]*(1+lam)
        dlm=np.dot(np.dot(gradf,-1),inv(hessf))
	print dlm
	print norm_grad(dlm)

        #maj de a
        a1=np.add(a0,dlm)
        #calcul du nouveau cout
        f1=f(x,y,a1[0],a1[1])

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
        norms.append(norm_grad(dlm))
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
a=[1.5,1.5]

a_final=levenberg(a,x,y,lam,nb_iter_max,norm_min)


plt.plot(np.log10(lambdas))
plt.show()
plt.plot(error)
plt.show()
plt.plot(norms)
plt.show()


