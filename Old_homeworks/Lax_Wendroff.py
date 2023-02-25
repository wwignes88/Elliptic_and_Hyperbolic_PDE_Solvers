

import numpy as np
import matplotlib.pyplot as plt


N    = 200    # Number of time steps
k    = 1/N     # time step size

M = 100
h = 1/M 

t0 = int(0.1/k)  # identify index correspondint to t = 0.1
t1 = int(0.2/k)
t2 = int(0.4/k)
t3 = int(0.6/k)
t4 = int(0.8/k)
t5 = int(1.0/k)

X = np.array([-1])
U = np.array([0])
i = -M
while i < M:
    x   = (i+1)*h
    val = np.exp(-16*(x**2))
    U   = np.append(U,val)
    X   = np.append(X,x)
    i   += 1
Un = np.copy(U)

#move forward in time
t  = 0
while t < N:
    i  = 1
    Ut = np.array([0])
    while i <= 2*M:
        if i <2*M:
            ui1= Un[i+1]
        if i == 2*M:
            ui1= 0
        ui1
        ui = Un[i]
        ui0= Un[i-1]

        a = (ui**2)*(ui1 - 2*ui + ui0)/(h**2)
        b = ui*( ((ui1-ui0)/(2*h))**2)
    
        c = k*ui*(ui1-ui0)/(2*h)

        uni = ui - c + 0.5*(k**2)*(b + a) 
        Ut  = np.append(Ut,uni)
        
        i   += 1
    
    U  = np.vstack((U,Ut))
    Un = np.copy(Ut)
    t += 1


# Plot calculated solution
plt.plot(X,U[t0,:],'blue',X,U[t1,:],'red',
         X,U[t2,:],'blue',X,U[t3,:], 'green',\
         X,U[t4,:], 'k--',X,U[t5,:],'r--')          
plt.xlabel('x')


plt.title('black: t=0,      red: t=0.2'+\
    '\nblue: t=0.4,      green: t=0.6' +\
    '\nblack--: t=0.8,      red--: t=1.0')



