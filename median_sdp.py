import numpy as np
import cvxopt as cvx 

import sys

import scipy.stats as stat 

import time

from pprint import pprint
import picos as pic

from math import ceil, floor


def Geometrical_median(Z,eps,max_iter = 10000):
    # We suppose y is    always distinct from the Z_1,...,Z_k
    # Weiszfeld algorithm
    k,d = Z.shape
    y=np.zeros(d)
    T=0
    dist = np.linalg.norm(Z,axis=1)
    while np.linalg.norm(((y-Z).T/dist).sum(1)) > eps and T<max_iter:
        
        norm = ( 1/dist ).sum()
        y = (Z.T/dist).sum(axis=1)
        y = y/norm
        dist = np.linalg.norm(Z-y,axis=1)
    return y


def Geometrical_MOM(X,k,eps=0.001):
    
    b = floor(n/k)
    Z = np.array([ X[i*b:(i+1)*b,:].mean(0)  for i in range(k)])
    return Geometrical_median(Z,eps)


# Gros morceau : SDP MT 


def MT_hopkins(x,r,Z):
    # implementation of SDP relaxation of MTE problem /!\ as found in Hopkins[2018]

    k,d = Z.shape
    Z_c = Z-x
    
    sdp = pic.Problem()
    z_c = pic.new_param('z_c',cvx.matrix(Z_c))
    X = sdp.add_variable('X', (1+k+d,1+k+d),vtype ="symmetric")

    sdp.add_constraint(X >> 0)
    sdp.add_constraint(X[0,0]==1)
    sdp.add_list_of_constraints([X[i,i]<=1 for i in range(1,k+1)])
    sdp.add_constraint(pic.trace(X[k+1:,k+1:]) <= 1 )
    sdp.add_list_of_constraints([(z_c[i,:].T| X[k+1:,i+1])  >=  r*X[0,i+1] for i in range(k)])
    sdp.set_objective('max',pic.sum([X[0,i] for i in range(1,k+1 )]))
    sdp.solve(verbose = 0,solver='cvxopt')
    return X.value,sdp.obj_value() 

    
def MT_chera(x,r,Z):
    # implementation of SDP relaxation of MTE problem into MT /!\ as found in Cherapanamjeri[2019]

    k,d = Z.shape
    Z_c = Z-x

    sdp = pic.Problem()
    z_c = pic.new_param('z_c',cvx.matrix(Z_c))
    X = sdp.add_variable('X', (1+k+d,1+k+d),vtype ="symmetric")
    sdp.add_constraint(X >> 0)   
    sdp.add_constraint(X[0,0]==1)
    sdp.add_list_of_constraints([X[i,i] - X[0,i] == 0 for i in range(1,k+1) ])
    sdp.add_constraint(pic.trace(X[k+1:,k+1:]) == 1 )
    sdp.add_list_of_constraints([(z_c[i,:].T| X[k+1:,i+1])  >=  r*X[0,i+1] for i in range(k)])
    sdp.set_objective('max',pic.sum([X[0,i] for i in range(1,k+1 )]))
    

    sdp.solve(verbose = 0,solver='cvxopt')
    return X.value,sdp.obj_value() 





# Si (un des) MT marche le reste devrait tourner...


def Distance_Estimation(Z,x,Hopkins=False,N=64):
    # find the maximum admissible d_t by dichotomic search
    
    r_range = np.power(2,np.linspace(-3,3.8,N+1))
    a= 0
    b = N
    while b-a>=2:
        m = int((a+b)/2)
        X,M = MT_chera(x,r_range[m],Z)
        if M>= 0.9:
            if m==N:
                return r_range[N]
            X,M = MT_chera(x,r_range[m+1],Z)
            if M<0.9:
                return r_range[m]
            a = m
        else:
            if m==0:
                return r_range[0]
            b = m  
    return r_range[m]




def Gradient_Estimation(Z,x):
    k,d = Z.shape
    d_star = Distance_Estimation(Z,x)
    X,m = MT_chera(x,d_star,Z) 
    X_v = X[k+1:,k+1:]
    v= np.linalg.svd(X_v)[2] # When using python 3 can add option hermitian=True
    g = v[0,:]
    H = np.where((Z-x).dot(g)>=0)[0]
    if H.size >= 0.9*k:
        return g
    else: 
        return -g





def Mean_Estimation(X,delta,T,gamma,record_files,mu=0,warm_start = False,shuffle= False ):
    #k = 3200*np.log(1/delta) We choose to not use the same value for k
    str_0 = str(mu_)+'\n'
    
    t=time.time()
    f,f_2,f_3,f_4 = record_files
    f.write(str_0)
    k = int(ceil(10*np.log(1/delta)))
    n,d = X.shape
    b = int(floor(n/k))
    Z = np.array([ X[i*b:(i+1)*b,:].mean(0)  for i in range(int(k))])
    if warm_start:
        x_star = Geometrical_median(Z,eps=0.001,max_iter = 10000)
        x_t = x_star
        d_t = Distance_Estimation(Z,x_t)
        d_star = d_t
    else :
        x_star,x_t = np.zeros(d),np.zeros(d)
        d_star,d_t =  np.Inf,np.Inf
    for i in range(T+1):
        # Z = np.array([ X[i*b:(i+1)*b,:].mean(0)  for i in range(k)]) ??
        if shuffle:
            np.random.shuffle(X) 
            Z = np.array([ X[i*b:(i+1)*b,:].mean(0)  for i in range(k)])
        d_t = Distance_Estimation(Z,x_t)
        g_t = Gradient_Estimation(Z,x_t)
        if d_t < d_star:
            x_star = x_t 
            d_star = d_t
        x_t = x_t +gamma*d_t*g_t
        loss = np.linalg.norm(x_t-mu)
        distance = d_t
        print("x",i, " = ",x_t," | d",i , " = ",d_t , " | g",i , " = ",g_t," | l",i, " = ",loss)
        
        strin = "x"+str(i)+ " = "+str(x_t)+" | d"+str(i)+ " = "+str(d_t)+ " | g"+str(i)+" = "+str(g_t)+ "loss : " + str(loss) + " \n "
        str_2 = str(loss)+'\n'
        str_3 = str(distance)+'\n'
        str_4 = str(time.time()-t) + '\n'
        f.write(strin)
        f_2.write(str_2)
        f_3.write(str_3)
        f_4.write(str_4)
    return x_star,d_star

#n,d = 1000,10
#delta = 0.01
sig = 1
n=1000 
delta_pow,d,dist = sys.argv[1:]
delta,d = 10**(-int(delta_pow)),int(d)

mu_ = stat.norm.rvs(d)
mu_ = 2*mu_/(np.linalg.norm(mu_))

#X_ = sig*np.random.randn(n,d) + np.array([1]*( d)) # We take mu so that ||\mu|| = 4
X_ = sig*stat.t.rvs(3,size=n*d).reshape(n,d) + mu_
eps = 0.01
T_star= -1000*np.log(eps)
T = 100
gam = 1/20


str_1 = 'student_d_'+str(d)+'_del_'+'_n_'+str(n)+'_rdvect_shuffle.txt'
str_2 = 'student_loss_d_'+str(d)+'_n_'+str(n)+'_rdvect_shuffle.txt'
str_3 = 'student_distance_d_'+str(d)+'_n_'+str(n)+'_rdvect_shuffle.txt'
str_4 = 'student_time_d_'+str(d)+'_n_'+str(n)+'_rdvect_shuffle.txt'
f=open(str_1,'w')
f_2= open(str_2,'w')
f_3= open(str_3,'w')
f_4= open(str_4,'w')

Mean_Estimation(X_,delta,T,gam,(f,f_2,f_3,f_4),mu=mu_,warm_start=False,shuffle=True)
f.close()
f_2.close()
f_3.close()
f_4.close()


str_1 = 'student_d_'+str(d)+'_n_'+str(n)+'_rdvect.txt'
str_2 = 'student_loss_d_'+str(d)+'_n_'+str(n)+'_rdvect.txt'
str_3 = 'student_distance_d_'+str(d)+'_n_'+str(n)+'_rdvect.txt'
str_4 = 'student_time_d_'+str(d)+'_n_'+str(n)+'_rdvect.txt'
f=open(str_1,'w')
f_2= open(str_2,'w')
f_3= open(str_3,'w')
f_4= open(str_4,'w')

Mean_Estimation(X_,delta,T,gam,(f,f_2,f_3,f_4),mu=mu_,warm_start=False,shuffle=False)

f.close()
f_2.close()
f_3.close()
f_4.close()

print("OK")