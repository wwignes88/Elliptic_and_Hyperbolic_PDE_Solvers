
# -*- 4+-
# coding: utf-8 -*-
"""104
Created on Sat Oct  3 15:39:47 2020

@author: wayne.wignes@wsu.edu
"""


import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
R    = 100
N    = 100    # Number of time steps
k    = 1/N    # time step size
M    = 20     # Grid length

N_A  = M**2   # length of matrix A
hT   = 2*pi/M # Theta step size
hR   = R/M    # r step size

V0          = 1 #initial speed

Nf          = 10      # number of time steps taken
tf          = Nf*k    # final time
Xview,Yview = 15,110  # Adjust plot view angle
lwidth      = 0.7

tol  = 0.001   # toleratnce parameter for Gauss siedell stopping criteria

rR  = k/(hR**2) # matrix parameter
rT  = k/(hT**2)


# Define initial boundary condition (polar coordinates)
def bndryP(r,theta):  
    return V0*(r**2-R**2)/4
# Define initial boundary condition (cartesian coordinates)
def bndryXY(x,y):  
    p  = np.sqrt(x**2 + y**2)
    return V0*(p**2-R**2)/4

# for polar to cartesian [plot] conversion
def fx(r,t):  
    return r*np.cos(t)
# for polar to cartesian [plot] conversion
def fy(r,t):  
    return r*np.sin(t)
######################################################
#######  Section #1: natural row ordered vector ######
######################################################

G     = np.zeros((M, M))    # Initiate grid-point matrix (for debugging)
K     = np.array([])         # Array to label grid points
X     = np.array([])         # Coordinate array
Y     = np.array([])  
RT    = np.array([0,0])      # Coordinate array


i  = 0
while i < M:
    j = 0
    while j < M:
        if j ==0:
            ri = hR/2
        else:
            ri = j*hR
        Ti = i*hT
        RT = np.vstack((RT,[ri,Ti]))

        kg = j + M*i          # natural row ordering 
        K  = np.append(K,kg)
        G[i,j] = kg          # Number grid points (debugging matrix)
        j += 1
    i += 1
RT    = np.delete(RT,0,0)
RAD   = RT[:,0]  # store polar array
THET  = RT[:,1]
RAD[0] = hR/3 # avoid r=0
#print("\nG = \n", G)  # print grid
      
# Define inital condition @ t=0
i = 0
U = np.array([])
while i < N_A:
    rt  = RT[i]
    r   = rt[0]
    t   = rt[1]
    U   = np.append(U,bndryP(r,t))  #initial vector
    i  += 1
u0 = np.copy(U)


#############################################
#######    Section #2                 #######
#######    Define Matrix operaturs     ######
#############################################
# Define Matrix A (for x derivative)
# Define Matrix B (for y derivative)
# Define Matrix AA (for debugging)
# Define Matrix bb (for debugging)

A   = np.zeros((N_A,N_A))   # Initiate matrix A
B   = np.zeros((N_A,N_A))   # Initiate matrix A
BB  = np.zeros((N_A,N_A))   # Debugging matrix
AA  = np.zeros((N_A,N_A))   # Debugging matrix

# Matrix for horixontal x derivatives
def matrixR(a,RADIAL):
    i = 0
    ri = RADIAL[i]
    while i < N_A :
        j = 0
        while j < N_A:
            if i==j:
                A[i,j]     =  a - (2 + hR/ri)*rR   # diagonals
                AA[i,j]    =  K[i]                   
                if i < N_A-1:       
                    A[i,j+1]    = (1 + hR/ri)*rR   # right hand neighbor
                    AA[i,j+1]   = K[i+1]
                if i > 0:                        
                    A[i,j-1]    = rR   # left hand neighbor
                    AA[i,j-1]   = K[i-1]
                if i == j and j%M == 0: #identify left boundaries on r,theta grid
                    ig = int(j/M)       # identify row (i) index of ghost point on grid
                    Ig = (ig +M/2)%M    # identify index of theta + pi (% is modulo  function)
                    kA = 0 + int(Ig*M)  # identify natural row index of new grid point (theta + pi, +r)
                    A[i,kA]  = rR
                    AA[i,kA] = 33       # debugging matrix
                    #print('kA = ', kA)
            j += 1
        i += 1

 
    # Erase boundary neighbor values from matrix A
    i = 1
    while i < N_A :
        j = 1
        while j < N_A :
            if j%M  == 0 and j%i ==1:  
                A[i,j]  =  0  # erase right-hand bndry nghbr.
                AA[i,j] =  0  
            if  i%M == 0 and i%j ==1:  
                A[i,j]  =  0 # erase left-hand bndry nghbr.
                AA[i,j] =  0  
            j += 1
        i += 1
    return A

# Matrix for vertical y derivatives
def matrixT(b,RADIAL):
    i = 0
    ri = RADIAL[i]
    while i < N_A :
        j = 0
        while j < N_A:
            if i==j:
                B[i,j]     =  b-2*rT/(ri**2)   # diagonals
                BB[i,j]    =  K[i]                   
                if i < M*(M-1):       
                    B[i,j+M]    = rT/(ri**2)    # right hand neighbor
                    BB[i,j+M]   = K[i+M]
                if i >= M:                        
                    B[i,j-M]    = rT/(ri**2)    # left hand neighbor
                    BB[i,j-M]   = K[i-M]
                if i == j and i < M :
                    kB = i + (M-1)*M
                    BB[i,kB] = 44
                    B[i,kB]  = rT
            j += 1
        i += 1
    return B
#A =  matrixR(0)
#print("U  = \n\n",U,"....Grid with natural row ordering")    # Debugging matrix (grid pts numbered)
#print("A = \n\n",AA,"computation matrix with grid points labeled")   # Debugging matrix (entries of A are numbered grid pts.)


def multiply(F,u): # Multiply matrix F and vector u
    i = 0
    V = np.array([])
    while i < len(F):
        a = np.sum(F[i,:]*u)
        V = np.append(V,a)
        i += 1
    return V

#  Convert 1-d solution vector into a 2-d grid (for plotting w/ meshgrid)
def Grid(vector):
    val    = np.sqrt(len(vector))
    length = int(val)   # square root of length of vector is grid length
    G      = np.zeros((length,length))
    j      = 0
    while j < length:
        i = 0
        while i < length:
            pt     = j+1 + length*i     # natural row numbering index
            l      = pt-1                # because python counts from zero
            G[i,j] = vector[l] # assign values on grid
            i     += 1
        j+=1 
    return G  

#############################################
#######  Section #3                       ##
#######  Define Guass siedell relaxation  ##
#############################################


def relax(matrix,Vin,fin, tol):
    norm  = tol + 0.1
    L     = len(Vin)
    kk    = 0
    while norm > tol:
        sumL = 0 # initial  lower matrix su,
        sumU = 0 # initiate upper matrix sum
        i    = 0 
        V0   = np.copy(Vin) 
        while i < L:
            aii  = matrix[i,i]
            if i > 0:
                sumL = np.sum(matrix[i,0:i]*Vin[0:i])   # lower sum
            if i < L-1:
                sumU = np.sum(matrix[i,i+1:]*Vin[i+1:]) # upper sum
            f  = fin[i]
            vi = (1/aii)*(f - sumL - sumU)   # solve for element
            Vin[i] = vi                      # redefine vector element
            i += 1
        norm1 = np.square(Vin-V0)
        norm2 = np.sum(norm1)
        norm  = np.sqrt(norm2)
        #print("\nnorm = ",norm)  # for debugging
        kk+=1
    print("\n# of relaxations  = ",kk)
    return Vin


#############################################
#######  Section #4                       ##
#######  Main loop: March forward in time ##
#############################################
f1 = multiply(matrixR(1,RAD),u0)
U1 = relax(-matrixT(-1,RAD),u0,f1,tol) 
f2 = U1 - multiply(matrixR(0,RAD),u0)
U2 = relax(-matrixR(-1,RAD),U1,f2,tol)

i = 0
while i < Nf: # Nf = number of desired time steps
    f1 = multiply(matrixR(1,RAD),u0)
    U1 = relax(-matrixT(-1,RAD),u0,f1,tol)   ####
    
    f2 = U1 - multiply(matrixT(0,RAD),u0)
    U2 = relax(-matrixT(-1,RAD),U1,f2,tol)
    
    U = np.vstack((U,U2))
    u0 = np.copy(u0)
    i +=1


################################################
#######     Section #5            ##############
#######     Plotting              ##############
################################################
        
# create polar arrays for plotting
radius  = np.linspace(0,    R-hR, M)
theta   = np.linspace(0, 2*pi-hT, M)

Rm,Tm   = np.meshgrid(radius,theta) #construct polar meshgrid

RADg    = Grid(RAD)      # Convert stored polar array to polar grid
THETg   = Grid(THET)

xg      = fx(RADg,THETg) # convert polar grid to cartesian grid
yg      = fy(RADg,THETg)

xe      = fx(Rm,Tm)      # convert meshgrids to cartesian
ye      = fy(Rm,Tm)

            
# Plot calculated solution
zc    = Grid(U2)  # Transform solution to meshgrid matrix
fig1  = plt.figure() 
ax    = plt.axes(projection='3d')
ax.plot_wireframe(xg, yg, zc, color='red',linewidths=lwidth)
ax.set_title('Calculated Solution @ time = '+str(Nf*k)+' \n\n' +\
             'R = '+str(R)+',      # time steps = '+str(Nf));
plt.xlabel('x')
plt.ylabel('y')
ax.view_init(Xview, Yview)

# Plot exact solution


ze     = bndryXY(xe, ye) # N*time = 2 (final time)

fig2   = plt.figure()
ax     = plt.axes(projection='3d')

#ax.plot_surface(xe, ye, ze, color='black') 
#my_cmap = plt.get_cmap('cool') 
#surf = ax.plot_surface(xe, ye, zc, cmap = my_cmap, edgecolor ='none') 

ax.plot_wireframe(xe, ye, ze, color='black',linewidths=lwidth)
ax.set_title('Solution @ time = 0.00 \n\n' +\
              'R = '+str(R));
plt.xlabel('x')
plt.ylabel('y')

ax.view_init(Xview, Yview)












