
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import numpy as np


# L = float(1.0)
# Re = float(1000)    # Reynolds number 
# N = 10		# mesh cells in x- and y-direction

# u = np.zeros((2*N*(N+1),1), dtype = np.float64)
# p = np.zeros((N*N+4*N,1), dtype = np.float64)
# tx = np.zeros((N+1,1), dtype = np.float64)     # grid points on primal grid
# x = np.zeros((N+2,1), dtype = np.float64)      # grid points on dual grid
# th = np.zeros((N), dtype = np.float64)       # mesh width primal grid
# h = np.zeros((N+1), dtype = np.float64)      # mesh width dual grid 

# #Generation of a non-uniform grid
# x[0] = 0
# x[N+1] = 1
# for i in range(N+1):
#     xi = i*L/N
#     tx[i] = 0.5*(1. - np.cos(np.pi*xi))     #  tx mesh point for primal mesh
#     if i>0:
#         th[i-1] = tx[i] - tx[i-1]           # th mesh width on primal mesh
#         x[i] = 0.5*(tx[i-1]+tx[i])          # x mesh points for dual mesh
        
# for i in range(N+1):
#     h[i] = x[i+1]-x[i]                      # h mesh width on dual mesh

# print("primal widths")
# print(th)
# print()
# print("dual widths")
# print(h)

# coord_stack_h = np.cumsum(np.insert(h,0,0))
# coord_stack_th = np.cumsum(np.insert(th,0,0))

# X_h, Y_h = np.meshgrid(coord_stack_h,coord_stack_h)

# X_th, Y_th = np.meshgrid(coord_stack_th,coord_stack_th)


def get_Ht02(primal_width_1d, N):
    cell_As = np.zeros(N*N)

    c = 0
    for j in primal_width_1d:
        for i in primal_width_1d:
            cell_As[c] = i * j
            
            c+=1
    H0t2 = sparse.diags(cell_As,format='csc')
    return splinalg.inv(H0t2)


def get_H1t1(primal_width_1d, dual_width_1d, N):
    h_e_mat = np.tile(np.tile(dual_width_1d,N),2)
    th_e_mat = np.tile(np.repeat(primal_width_1d,N+1),2)

    e_ratio = h_e_mat/th_e_mat

    return sparse.diags(e_ratio,format='csc')

# plt.pcolormesh(X_th, Y_th,cell_As.reshape(N, N), edgecolors='k', linewidths=0.2)

# plt.pcolormesh(X_h, Y_h,C_h, edgecolors='k', linewidths=2, cmap = 'binary')

# plt.legend()


# plt.show()

# plt.imshow(H0t2.todense())
# plt.colorbar()
# plt.show()

# plt.imshow(H1t1.todense())
# plt.colorbar()
# plt.show()