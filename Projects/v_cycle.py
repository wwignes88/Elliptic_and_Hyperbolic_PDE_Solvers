"""
v-cycle method. By Wayne Wignes
"""
import numpy as np
import matplotlib.pyplot as plt

h = (1/6)**2
pi  = np.pi 
M = 8   
s = 0.4  #Weighted Jacobi correction factor

def f(M,j):              # fine grid solution (zero)
    W = np.array([])
    k = 1
    while k < M:
        val = 0
        W   = np.append(W,val)
        k += 1
    return W 
def eigR(M,j):          # eigenvalue of R 
    a = (np.cos(j*pi/(M)))**2 
    b = 1 - 2*s*a 
    return b
def eigA(M,j):          # eigenvalue for matrix A
    a = np.cos(j*pi/(M))
    b = 1/((8/M)**2)    # course grid correction factor
    return 2*b*(1-a)     
def w(M,j):             # Eigenvector of matrix A
    W = np.array([])
    k = 1
    while k < M:
        val = np.sin(j*pi/M)
        W   = np.append(W,val)
        k  += 1
    return W    
def u(M,j,n):           # approximation after first n relaxations (eqn. [])
    a = eigR(M,j)
    return (a**n)*w(M,j)   
def rh(M,j,n,w):        # residual on fine grid (eqn. [])
    a  = eigR(M,j)  
    b  = eigA(M,j)
    return -(a**n)*b*w  # w(M,j) 
rh_v = np.vectorize(rh) # this function will accept vector w
def r2h(M,j,n,w):       #  course grid residual (eqn. [])
    a = eigR(M,j)    
    b = eigA(M,j)
    c = np.cos(j*pi/(2*M))
    return -(a**n)*b*(c**2)*w  # w(M/2,M-j) 
r2h_v = np.vectorize(r2h) # this function will accept vector w


######################################################
######## Define restrict and       ###################
######## prolongate matrices'      ###################
######################################################
    
# Prolongation operators
Ph = 0.5*np.zeros((M-1, 3))   # 2h -> h
Ph[0,:] = [1,0,0]
Ph[1,:] = [2,0,0]
Ph[2,:] = [1,1,0]
Ph[3,:] = [0,2,0]
Ph[4,:] = [0,1,1]
Ph[5,:] = [0,0,2]
Ph[6,:] = [0,0,1]

P2h    = 0.5*np.zeros((3,1))     # 4h -> 2h
P2h[0] = 1
P2h[1] = 2
P2h[2] = 1
    
# Restriction operators
R2h      = 0.25*np.zeros((3, M-1))  # h -> 2h
R2h[0,:] = [1,2,1,0,0,0,0]
R2h[1,:] = [0,0,1,2,1,0,0]
R2h[2,:] = [0,0,0,0,1,2,1]

R4h = 0.25*np.array([1,2,1])              # 2h -> 4h



######################################################
######## Define Au = f relaxation  ###################
######## for fine grid (h)         ###################
######################################################


def RELAX(M,j,n,u,F):
        
    ###################################
    ######## Create matrix A_h ########
    ######## for fine grid     ########
    ###################################  
    A   = np.zeros((M+1,M+1))
    cgf = 1/((8/M)**2)  #course grid correction factor
    ii   = 1
    while ii < M:
        jj = 1
        while jj < M:    
            if ii == jj :
                A[ii,jj]    =  2/h  
                A[ii,jj+1]  = -1/h
                A[ii,jj-1]  = -1/h 
            jj += 1
        ii += 1  
    # delete extra rows and columns
    A1 = np.delete(A,[0,M],0) 
    A = np.delete(A1,[0,M],1)      
    A = cgf*A
    #print("\nA = \n", A)
    
    ##########################################
    ######## Weighted Jacobi relaxation ######
    ##########################################
    
    R = np.array([])
    s  = 0.4 # Define relaxation parameter 
    k = 0
    N    = np.array([])
    DIFF = np.array([])
    R    = np.array([])
    while k < n:
        Uk   = np.array([])
        i    = 0
        while i < M-1:
            if i==0:    # Define first element of S_K+1
                sumL  = np.sum(A[0,0:M-1]*u[0:M-1])
                a     = (F[0] - sumL)/A[0,0]
                Uk1   = (1-s)*u[0] + s*a
                Uk    = np.append(Uk,Uk1)
            if i>0:
                sumL  = np.sum(A[i,0:i-1]*u[0:i-1]) # Lower triangular Mat.
                sumU  = np.sum(A[i,i+1:M-1]*u[i+1:M-1]) # Upper '  ' 
                b     = (F[i] - sumL - sumU)/A[i,i]
                Uki   = (1-s)*u[i] + s*b  # define S_k+1(i)
                Uk    = np.append(Uk,Uki)       
            i += 1 
        #print("rk"  , rk)
        # calculate norms:
            
        # ||u_k+1 - u_k ||
        diffval  = Uk - u
        DIFFval  = np.sqrt( np.sum( np.square(diffval)))
        DIFF     = np.append(DIFF,DIFFval)
        
        # || f - Au_k ||
        i = 0
        rk   = np.array([])
        while i < M-1:
            rval = F[i] - np.sum(A[i,:]*Uk)
            rk   = np.append(rk,rval)
            i   += 1
        Rval = np.sqrt( np.sum( np.square(rk)))
        R    = np.append(R,Rval)     
        
        u  = Uk    # set u_k = u_k+1
        N  = np.append(N,k)  # vector for plotting purposes
        k += 1
    return [Uk,rk,R,DIFF,N] # return u_k and residual vector



######################################################
################### V-cycle    #######################
######################################################

M  = 8
M2 = 4
n  = 60
m  = n

def V_cycle (n,m,j):
    fh = f(M,j) # fine grid solution (=0)

    # Make the inital guess uh   = w(M,j)
    # With f=0 on the fine grid we can use eqn. [] instead of solving
    # for u after n relaxations, 
    uh_n = u(M,j,n)  
  
    # With f=0 on the fine grid there is no need to do a grid transfer
    # we can instead use eqn. [3.2] and [3.3] directly,
    rh  = rh_v(M,j,n,w(M,j))       # equation [3.2]
    r2h = r2h_v(M,j,n,w(M/2,M-j))  # equation [3.3]

    # Use r2h to solve Ae=r on course grid with initial guess e=0
    e2       = np.zeros((M2-1))
    relax1   = RELAX(M2,j,n,e2,r2h) # relax n times on fine grid
    e2_n     = relax1[0]  # course grid (G2) error after n relaxations
    res2_n   = relax1[1]  # course grid (G2) residual after n relaxations
    norm1_n  = relax1[2]  # vector-norm of n residual(r_k) to be plotted
    norm2_n  = relax1[3]  # vector-norm of n difference values; |u_k+1-u_k|
    n_plot   = relax1[4]  # vector of indices' for plotting purposes

    # calculate residual on coursest grid (G4) using restriction operator
    r4_n = np.sum(R4h*res2_n) 
    # can simply divide by matrix A because is single valued
    A4   = 1/(8*h)  # eqn. [1.7]
    e4_n = r4_n/A4

    # Correct error on G2 grid by prolongation of G4 error
    e2_n = e2_n + np.sum(P2h*e4_n)     
        
    # Correct un with prolongation of e2_n -the error accumulated  
    # through the grid transfers thus far     
    i = 0
    u_transfer = np.array([])
    while i<M-1:
        val = np.sum(Ph[i,:]*e2_n)
        u_transfer = np.append(u_transfer,val)
        i += 1
    uh_n = uh_n + u_transfer      
        
    # Solve Au = f on fine grid with initial guess uh_n
    relax3   = RELAX(M,j,m,uh_n,fh) # relax m times on fine grid
    uh_m     = relax3[0]  # fine grid (G1) solution after m relaxations
    res_m    = relax3[1]  # fine grid (G1) residual after m relaxations        
    norm1_m  = relax3[2]  # vector-norm of n residual(r_k) to be plotted
    norm2_m  = relax3[3]  # vector-norm of n difference values; |u_k+1-u_k|
    m_plot   = relax3[4]  # vector of indices' for plotting purposes

    return[uh_m, n_plot, norm1_n, m_plot, norm1_m]
V_cycle_v = np.vectorize(V_cycle)

# run V-cycle for j=1,2,3
j      = 1
ua     = 0
norm_na = 0
norm_ma = 0
while j<=3:
    Vcyca     = V_cycle(n,m,j)
    ua       += Vcyca[0] # add u (solution of Au=f) vectors for j = 1,2,3
    norm_na  += Vcyca[2] # add normal || r || vectors for j=1,2,3
    norm_ma  += Vcyca[4] # add normal || r || vectors for j=1,2,3
    N_plot    = Vcyca[1] # index vetor for ploting n relaxations
    M_plot    = Vcyca[3] # index vetor for ploting m relaxations
    j += 1   

# run V-cycle for j=4
j=4
Vcycb    = V_cycle(n,m,j)
ub       = Vcycb[0] # add u (solution of Au=f) vectors for j = 1,2,3
norm_nb  = Vcycb[2]   
norm_mb  = Vcyca[4] 

# run V-cycle for j=5,6,7               
j      = 5
uc     = 0
norm_nc = 0 
norm_mc = 0
while j<8:
    Vcycc     = V_cycle(n,m,j)
    uc       += Vcycc[0] # add u (solution of Au=f) vectors for j = 5,6,7  
    norm_nc  += Vcycc[2] # add normal || r || vectors for j=5,6,7 (n relaxations)
    norm_mc  += Vcyca[4] # add normal || r || vectors for j=5,6,7 (m relaxations)
    j += 1   
               
# Calculate final norm and u vectors,

u      = ua + ub + uc       
norm_n = norm_na + norm_nb + norm_nc  # Final norm for j=1,2,...,7 (n iterations)
norm_m = norm_ma + norm_mb + norm_mc  # Final norm for j=1,2,...,7 (m iterations)
     
plt.figure(1)          
plt.plot(N_plot,norm_n,'b',  M_plot,norm_m,'g')
plt.title(" || r || = || f-Au ||\n \
              blue: residual for first (n) relaxation\n \
            green: residual for second (m) relaxation ")   
plt.axis([0,max(N_plot),0,max(norm_m)/3])       
        
        
        
        
