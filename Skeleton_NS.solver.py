# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:25:58 2021

@author: mgerritsma
"""


from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import numpy as np
import incidence as inc
import hodge as hod

#  00D#MMXXI#

# Determine a proper value for the tol which determines when the program terminates

tol = 1e-4
one = 1
mone = -1

L = 1.0
Re = float(1000)    # Reynolds number 
N = 32  		# mesh cells in x- and y-direction

u = np.zeros([2*N*(N+1),1])
p = np.zeros([N*N+4*N,1])
tx = np.zeros([N+1,1])     # grid points on primal grid
x = np.zeros([N+2,1])      # grid points on dual grid
th = np.zeros([N])       # mesh width primal grid
h = np.zeros([N+1])      # mesh width dual grid 

#Generation of a non-uniform grid
x[0] = 0
x[N+1] = 1
for i in range(N+1):
    xi = i*L/N
    tx[i] = 0.5*(1. - np.cos(np.pi*xi))     #  tx mesh point for primal mesh
    if i>0:
        th[i-1] = tx[i] - tx[i-1]           # th mesh width on primal mesh
        x[i] = 0.5*(tx[i-1]+tx[i])          # x mesh points for dual mesh
        
for i in range(N+1):
    h[i] = x[i+1]-x[i]                      # h mesh width on dual mesh
    
        
th_min = min(th)
h_min = min(h)
h_min = min(h_min,th_min)                   # determination smallest mesh size

dt = min(h_min,0.5*Re*h_min**2)             # dt for stable integration

#
#  Note that the time step is a bit conservative so it may be useful to see
#  if the time step can be slightly increased. This will speed up the
#  calculation.
# 

#  Boundary conditions for the lid driven acvity test case
U_wall_top = -1
U_wall_bot = 0
U_wall_left = 0
U_wall_right = 0
V_wall_top = 0
V_wall_bot = 0
V_wall_left = 0
V_wall_right = 0


# Set up the sparse incidence matrix tE21. Use the orientations described
# in the assignment.
# Make sure to sue sparse matrices to avoid memory problems

tE21, tE21_norm = inc.computetE21(N)

# Setup u_norm
LB = U_wall_left*np.ones(N) * th 
RB = U_wall_right*np.ones(N) * th
TB = V_wall_top*np.ones(N) * th
BB = V_wall_bot*np.ones(N) * th
u_norm = np.concatenate((LB, RB, BB, TB), axis=0)[:,np.newaxis]
# print('u_norm_shape',u_norm.shape)
# print('tE21_norm shape', tE21_norm.shape)
u_norm = tE21_norm @ u_norm


# Insert the normal boundary conditions and split of the vector u_norm

# Vector of fluxes is just zero...


# Set up the outer-oriented incidence matrix tE10
# Done

#  Set up the sparse, inner-oriented  incidence matrix E10
E10 = - tE21.transpose()
#   DONE

#  Set up the (extended) sparse, inner-oriented incidence matrix E21
E21, E21_norm = inc.compute_dual_E21(N)
#   DONE

tE10 = E21.transpose()


#  Split off the prescribed tangential velocity and store this in 
#  the vector u_pres
LB = V_wall_left*np.ones(N+1) * h 
RB = V_wall_right*np.ones(N+1) * h
TB = U_wall_top*np.ones(N+1) * h
BB = U_wall_bot*np.ones(N+1) * h
u_pres = np.concatenate((LB, RB, BB, TB), axis=0)[:,np.newaxis]
# print('u_pres shape', u_pres.shape)
# print('E21_norm shape', E21_norm.shape)
u_pres = E21_norm @ u_pres

#  Set up the Hodge matrices Ht11 and H1t1
Ht11 = hod.get_Ht11(th, h, N)
H1t1 = splinalg.inv(Ht11)

#  Set up the Hodge matrix Ht02
Ht02, _ = hod.get_Ht02(h, N)


A = tE21@Ht11@E10

n = A.shape[0]
LU = splinalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition


# print(Ht02.shape)
u_pres_vort = Ht02@u_pres
temp = H1t1@tE10@Ht02@u_pres 

u_pres = temp

VLaplace = H1t1@tE10@Ht02@E21
DIV = tE21@Ht11

ux_xi = np.zeros([(N+1),(N+1)], dtype = float)
uy_xi = np.zeros([(N+1),(N+1)], dtype = float)
convective = np.zeros([2*N*(N+1),1], dtype = float)

diff = 1
step = 0

while (diff>tol):
    
    xi = Ht02@E21@u + u_pres_vort
    
    ux_xi[:, 0] = U_wall_bot*xi[:(N+1),0]
    uy_xi[:, 0] = V_wall_left*xi[::(N+1),0]

    ux_xi[:, N] = U_wall_top*xi[N*(N+1):(N+1)*(N+1), 0]
    uy_xi[:, N] = V_wall_right*xi[N::(N+1), 0]

    ux_xi[:, 1:N] = np.reshape((u[(N+1):N*(N+1)]+u[:(N-1)*(N+1)])*xi[(N+1):N*(N+1)], ((N+1), (N-1)), order='f')/(2*h[:,None])
    uy_xi[:, 1:N] = np.reshape((u[N*(N+1):2*N*(N+1)]+u[N*(N+1)-1:2*N*(N+1)-1]), ((N+1), (N)), order='C')[:, 1:]*np.reshape(xi, (N+1, N+1))[:, 1:N]/(2*h[:,None]) 

    convective[:N*(N+1)] = np.reshape(-(uy_xi[:-1]+uy_xi[1:])*h/2, ((N*(N+1), 1)))
    convective[N*(N+1):2*N*(N+1)] = np.reshape((ux_xi[:-1]+ux_xi[1:])*h/2, ((N*(N+1), 1)), order='f')
            
    # Set up the right hand side for the equation for the pressure
            
    rhs_Pois = DIV@( u/dt - convective - VLaplace@u/Re - u_pres/Re) + u_norm/dt
    
    # Solve for the pressure
    
    p = LU.solve(rhs_Pois)
    
    # Store the velocity from the previous time level in the vector uold
    
    uold = u
    
    # Update the velocity field
    
    u = u - dt*(convective + E10@p + (VLaplace@u)/Re + u_pres/Re)
    # print(u.shape)
    # Every other 1000 iations check whether you approach steady state and 
    # check whether you satsify conservation of mass. The largest rate at whci 
    # mass is created ot destroyed is denoted my 'maxdiv'. This number should
    # be close to machine precision.
    
    if ((step % 1000)==0):
        maxdiv = np.max(np.abs(DIV@u+u_norm))
        diff = np.max(np.abs(u-uold))/dt
    
        print("maxdiv : ",maxdiv)
        print("diff   : ", diff)
         
       
    step += 1

    if step % 100==0: print("step at:",step)

print("steps taken:", step) 

def get_pressure(p, N):
    p_top = p[0:N]
    p_remaining = p[N:]

    p_bottom = p[len(p)- N:]
    p_remaining = p_remaining[0:len(p_remaining)-N]
    p_remaining = p_remaining.reshape(N,N+2)

    p_inner = p_remaining[:, 1:len(p_remaining[0]) - 1]
    return p_inner

p_inner = get_pressure(p, N)
print("Pressure inner:", p_inner)
plt.imshow(p_inner)
plt.show()


                    
# u that we have calculated is actually a circulation along edges on the dual grid
# We need to convert this to fluxes on the primal grid
# Thus we need to multiply with the Hodge matrix Ht11

print(f"N = {N}")
fluxes = Ht11@u

plt.imshow((E21 @ u).reshape((N+1,N+1)))
plt.title("U")
plt.show()

print(fluxes.shape)
print(len(tx)**2)
# print(fluxes)
print(E21.shape)
plt.imshow((E21 @ fluxes).reshape((N+1,N+1)))
plt.show()

# print(flux_curl.shape)



# print("fluxes shape", fluxes.shape)
