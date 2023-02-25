
import numpy as np
import matplotlib.pyplot as plt

N = 2000
  # number of time steps
M = 100   # number of spatial steps

L = 2
Y = .9# Tuning parameter - holding nonlinear term constant ( 0<Y<1)
h = L/M*2       # spatial step size
k = Y*(h**2)/(2-0.5*Y*(h**2))  
r = k/(h**2)

def bndry(x):     # inital value for solution to second eqn.; u2(x,t=0)
    if np.abs(x)<=1:
        return 0.1
    else:
        return 0

v = -(1/12)*np.sqrt(5/3)    #calculated parameters in exact solution
wk = -np.sqrt(1/15)

def exact (m,n):  # exact solution (for comparison)
    x = m*h
    t = n*k
    T = np.tanh(wk*(x-v*t))
    l = 1/30
    ARG = l*(3 - T + 12*(T**2)) 
    return ARG

   
A = np.zeros((2*M+2, 2*M+2)) # initial M+1 by M+1 matrix

i = 1
while i <= 2*M:    # define matrix A
    j = 1
    while j <= 2*M:
        if i==j :
            A[i,j-1] = r
            A[i,j+1] = r
            A[i,j]   = 1 - 2*r  
        j += 1
    i += 1
#print(A,"....Matrix A with extra 2 extra rows and columns\n")
a = np.delete(A,[0,2*M+1],0)   # delete first rows and columns (because boundary values are given)
#print(a, ".... delete first coloumn and row\n")
b = np.delete(a,[0,2*M+1],1)   # ... now is an M-1 by M-1 matrix
A = b  
#print("\n",A,"...... delete last column and row \n")
A[0,1] = 2*r   # Apply derivative boundary condition
A[0,0] = 1-2*r
#print(A," ......... initial conditoins; change first row entries")


EXACT  = np.array([]) 
U  = np.array([]) 
X  = np.array([])        
m  = -M
while m < M: 
    x = m*h
    U = np.append(U, bndry(x))    # assigning initial values for t=0
    EXACT = np.append(EXACT, exact(m,0)) # calculate exact solution
    X = np.append(X,x)            # vector for plotting purposes    
    m+= 1
u  = np.copy(U)     # inital solution vector


# This loop adds a [row] solution vector for every time value (or every n value)
n   = 1     # start at n=1 because n=0 was already calculated
while n < N:
    U_n = np.array([])
    EXACT_n = np.array([])
    m   = 0
    while m < 2*M:       # for every n value, enumerate M-1 rows of matrix A 
        a    = np.sum(A[:,m]*u) 
        b    = u[m]
        
        c    = k*b*(1-b)*(b-1/4)   #Forward Euler
        d    = a + c
        U_n  = np.append(U_n, d)
        mx   = m-M
        EXACT_n = np.append(EXACT_n, exact(mx,n))
        m    += 1
    u   = np.copy(U_n)          # set u1 equal to newly defined vector U1_n+1
    U   = np.vstack((U,U_n))    # append U1_n+1 to matrix U1 
    EXACT   = np.vstack((EXACT,EXACT_n))
    n  += 1


U0 = U[0,:]
Umid1 = U[500,:]
Umid2 = U[1000,:]
UN = U[N-1,:]


deltat=1000
v1 = (UN[2*M-1]-U0[2*M-1])/deltat
v2 = (UN[M-1]-U0[M-1])/deltat
print("\nv1 = ",v1)
print("\nv2 = ",v1)


plt.figure(1)    
plt.title("red = u(t=0) \n  \
          green = u(t=500*k) \n  \
          yellow = u(t=1000*k) \n  \
          blue = u(t=2000*k) ")
plt.xlabel('x')
plt.ylabel('u (x,t)')
plt.plot(X,UN,'b',X,U0,'r',X,Umid1,'g',X,Umid2,'y')       # plot solution for final time

#EN = EXACT[N-1,:]
#E0 = EXACT[0,:]

#plt.figure(2)    
#plt.title("EXACT  \nt = 800*k \n \
 #         red = u(t=0)   \
 #         blue = u(t=Nk) ")
#plt.xlabel('x')
#plt.ylabel('u (x,t)')
#plt.plot(X,EN,'b*',X,E0,'r*')       # plot solution for final time

