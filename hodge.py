
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
from scipy.sparse import diags
import numpy as np




def get_Ht02(h, N):

    diag = np.empty((N+1)**2)

    for d1, d1_val in enumerate(h):
        for d2, d2_val in enumerate(h):
            diag[d1*(N+1) + d2] = d1_val*d2_val

    H2t0 = diags(diag)
    Ht02 = diags(1/diag)

    return Ht02, H2t0, diag

def get_Ht11(primal_width_1d, dual_width_1d, N):
    # h_e_mat = np.tile(np.tile(dual_width_1d,N),2)
    # th_e_mat = np.tile(np.repeat(primal_width_1d,N+1),2)

    # # e_ratio = h_e_mat / th_e_mat
    # e_ratio = th_e_mat / h_e_mat
    h = dual_width_1d
    th = primal_width_1d
    diag = np.empty(2*N*(N+1))
    # firstly go through u fluxes
    for p, p_val in enumerate(th):
        for d, d_val in enumerate(h):
            diag[p*(N+1) + d] = p_val / d_val

    # secondly go through v fluxes
    for d, d_val in enumerate(h):
        for p, p_val in enumerate(th):
            diag[N*(N+1) + d*N + p] = p_val / d_val


    Ht11 = diags(diag)
    H1t1 = diags(1/diag)
    return H1t1, Ht11



def gen_plots():


    import shutil

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsfonts}'
    })

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

    # print("primal widths")
    # print(th)
    # print()
    # print("dual widths")
    # print(h)

    coord_stack_h = np.cumsum(np.insert(h,0,0))
    coord_stack_th = np.cumsum(np.insert(th,0,0))

    X_h, Y_h = np.meshgrid(coord_stack_h,coord_stack_h)

    X_th, Y_th = np.meshgrid(coord_stack_th,coord_stack_th)


    Ht02, H2t0, areas = get_Ht02(h,N)
    H1t1, Ht11 = get_Ht11(th, h, N)


    plt.title("DOF ratio from dual points to primal surfaces  $\mathbb{H}^{2\tilde{0}}$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.pcolormesh(X_h, Y_h,areas.reshape(N+1, N+1), edgecolors='k', linewidths=0.2)
    plt.colorbar()
    # plt.legend()
    # plt.savefig()
    plt.tight_layout()
    plt.savefig(f"images/hodge/h2t0_areas_N_{N}.pdf")
    # plt.show()
    plt.close()

    # plt.clf()
    plt.matshow(Ht02.todense())
    plt.title(r"$\mathbb{H}^{\tilde{0} 2}$")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"images/hodge/Ht02_N_{N}.pdf")
    # plt.show()
    plt.close()


    plt.matshow(H1t1.todense())
    plt.title(r"$\mathbb{H}^{1 \tilde{1}}$")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"images/hodge/H1t1_N_{N}.pdf")
    plt.close()

    plt.matshow(Ht11.todense())
    plt.title(r"$\mathbb{H}^{ \tilde{1} 1}$")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"images/hodge/Ht11_N_{N}.pdf")
    plt.close()


def main():
    gen_plots()


if __name__ == '__main__':
    main()