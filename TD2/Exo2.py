# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def g(x, a):
    return x**a[0] * np.exp(-a[1] * x)

def f(x, y, a):
    return 0.5 * np.sum((y-g(x,a))**2)

def grad_g_a1(x, a):
    return np.log(x)*x**a[0] * np.exp(-a[1] * x)

def grad_g_a2(x, a):
    return -x**a[0] * x * np.exp(-a[1] * x)
    #return -x**(a[0]+1) * np.exp(-a[1] * x)

def grad_f(x, y, a):
    return np.array([-np.sum((y - g(x, a)) * grad_g_a1(x, a)), -np.sum((y - g(x, a)) * grad_g_a2(x, a))])

def hess_f(x, a):
    return np.array([[np.sum(grad_g_a1(x, a)**2), np.sum(grad_g_a1(x, a) * grad_g_a2(x, a))],[np.sum(grad_g_a1(x, a) * grad_g_a2(x, a)), np.sum(grad_g_a2(x, a)**2)]])

def norm_grad(a):
    return np.sum(np.sqrt(np.array(a)**2))

a_jeu = np.array([2, 3])
b = 0.1

#Plot g(x, a)
X = np.linspace(0.01, 5, 500)
plt.plot(X, g(X, a_jeu))

#Creation et plot du jeu de donné
Y = g(X,a_jeu) + b * (np.random.randn(500))
plt.plot(X, Y, '+')

#TEST
#plt.plot(X, g(X, [2, 2]))
#plt.plot(X, g(X, [3, 2]))

plt.legend(loc='upper left')
plt.show()

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
	print "gradient de f: ", gradf
	hessf=hess_f(x,a0)
	#hessf[0,0] = hessf[0,0] * (1+lam)
	#hessf[1,1] = hessf[1,1] * (1+lam)
	print "hessienne :", hessf
	print "inverse de la hessienne :", np.linalg.inv(hessf)
	dlm=-np.dot(gradf, np.linalg.inv(hessf))
	#dlm = -gradf
	print "dlm :", dlm
	#maj de a
	a1=a0+dlm
	print "a", a1
	#calcul du nouveau cout
	f1=f(x,y,a1)
	print "nouveau cout (f) :", f1
	#si on descend : on continue en Newton (baisse de lambda)
	#if f1<f0:
	f0=f1
	a0=a1
	lam=lam/10
	#si on remonte : pas bon, on ne garde pas cette iteration et on tend + vers la descente du gradient (augmentation de lambda)
	#else:
	 #  lam=lam*10
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
    a1 = [aa[i][0] for i in range(len(aa))]
    a2 = [aa[i][1] for i in range(len(aa))]
    
    
    #Comparaison de la fonction avec le a donné puis apres convergence
    plt.plot(X, g(X, a), '-', label='a = (1.5, 1.5)')
    plt.plot(X, g(X, a0), '-', label='g(x,a) apres optimisation de a')
    plt.plot(X, Y, '+')
    plt.legend()
    plt.show()
    
    #Plot de l'evolution des differents paramètres
    plt.subplot(221)
    plt.plot(error, '-', label='fonction de cout')
    plt.legend()
    plt.subplot(222)
    plt.plot(a1, '-', label='a1')
    plt.plot(a2, '-', label='a2')
    plt.legend()
    plt.subplot(223)
    plt.plot(lambdas, label='lambda')
    plt.legend()
    plt.subplot(224)
    plt.plot(norms, label='Norme du gradient')
    plt.legend()
    
    plt.show()
    
    
    
    return a0


nb_iter_max=100
norm_min=1e-5
lam=0.001

a_init =  [1.5, 1.5]
levenberg(a_init, X, Y, lam, nb_iter_max, norm_min)


#TESTS
'''
print f(X,Y,a)
print grad_g_a1(2, a)
print grad_g_a2(2, a)
print grad_f(X, Y, a)
print hess_f(X, a)
'''
