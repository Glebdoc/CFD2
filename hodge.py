
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import numpy as np


def get_Ht02(dual_width_1d, N):
    Np1 = N+1
    cell_As = np.zeros(Np1*Np1)

    c = 0
    for j in dual_width_1d:
        for i in dual_width_1d:
            cell_As[c] = 1/(i * j)
            
            c+=1

    Ht02 = sparse.diags(cell_As,format='csc')

    return Ht02, cell_As


def get_Ht11(primal_width_1d, dual_width_1d, N):
    h_e_mat = np.tile(np.tile(dual_width_1d,N),2)
    th_e_mat = np.tile(np.repeat(primal_width_1d,N+1),2)

    # e_ratio = h_e_mat / th_e_mat
    e_ratio = th_e_mat / h_e_mat

    return np.flip(sparse.diags(e_ratio,format='csc'))

def main():

    import matplotlib.pyplot as plt

    L = float(1.0)
    N = 3		# mesh cells in x- and y-direction

    tx = np.zeros((N+1,1), dtype = np.float64)     # grid points on primal grid
    x = np.zeros((N+2,1), dtype = np.float64)      # grid points on dual grid
    th = np.zeros((N), dtype = np.float64)       # mesh width primal grid
    h = np.zeros((N+1), dtype = np.float64)      # mesh width dual grid 

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

    print("primal widths")
    print(th)
    print()
    print("dual widths")
    print(h)

    coord_stack_h = np.cumsum(np.insert(h,0,0))
    coord_stack_th = np.cumsum(np.insert(th,0,0))

    X_h, Y_h = np.meshgrid(coord_stack_h,coord_stack_h)

    X_th, Y_th = np.meshgrid(coord_stack_th,coord_stack_th)


    H0t2, cell_As = get_Ht02(h,N)
    H1t1 = get_Ht11(th, h, N)

    plt.pcolormesh(X_h, Y_h,cell_As.reshape(N+1, N+1), edgecolors='k', linewidths=0.2)

    # plt.legend()

    plt.show()

    plt.imshow(H0t2.todense())
    plt.colorbar()
    plt.show()

    plt.imshow(H1t1.todense())
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()