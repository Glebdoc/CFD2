
from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import numpy as np
import incidence as inc
import hodge as hod
import json
# import tqdm

#  00D#MMXXI#

# Determine a proper value for the tol which determines when the program terminates
def run_cavity_2d(N, tol, Re, print_status = False):
    tol = tol
    one = 1
    mone = -1

    L = 1.0
    Re = Re    # Reynolds number 
    N = N  		# mesh cells in x- and y-direction

    u = np.zeros([2*N*(N+1),1])
    p = np.zeros([N*N+4*N,1])
    tx = np.zeros(N+1)     # grid points on primal grid
    x = np.zeros(N+2)      # grid points on dual grid
    th = np.zeros(N)       # mesh width primal grid
    h = np.zeros(N+1)      # mesh width dual grid 

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
    u_norm = np.concatenate((LB, RB, BB, TB), axis=0)

    # Insert the normal boundary conditions and split of the vector u_norm
    u_norm = tE21_norm @ u_norm
    u_norm = u_norm[:,np.newaxis]


    E10 = -tE21.transpose()

    E21, E21_norm = inc.compute_dual_E21(N)
    tE10 = E21.transpose()


    #  Split off the prescribed tangential velocity and store this in 
    #  the vector u_pres
    LB = V_wall_left*np.ones(N+1) * h 
    RB = V_wall_right*np.ones(N+1) * h
    BB = U_wall_bot*np.ones(N+1) * h
    TB = U_wall_top*np.ones(N+1) * h

    u_pres = np.concatenate((LB, RB, BB, TB), axis=0)

    u_pres = (E21_norm @ u_pres)[:,np.newaxis]


    #  Set up the Hodge matrices Ht11 and H1t1
    H1t1, Ht11= hod.get_Ht11(th, h, N)

    # Ht11 = hod.get_Ht11(th, h, N)
    # H1t1 = splinalg.inv(Ht11)

    #  Set up the Hodge matrix Ht02
    #Ht02, _ = hod.get_Ht02(h, N)
    Ht02, H2t0, _  = hod.get_Ht02(h, N)

    A = tE21@Ht11@E10

    LU = splinalg.splu(A,diag_pivot_thresh=0) # sparse LU decomposition

    # print(Ht02.shape)
    temp = H1t1@tE10@Ht02@u_pres 

    VLaplace = H1t1@tE10
    DIV = tE21@Ht11

    ux_xi = np.zeros([N+1,N+1], dtype = float)
    uy_xi = np.zeros([N+1,N+1], dtype = float)
    convective = np.zeros([2*N*(N+1),1], dtype = float)

    diff = 1
    maxdiv = 1
    step = 1
    residual_step_hist = [step]

    maxdiv_list = [maxdiv]
    diff_list = [diff]


    while (diff>tol):
        
        xi = Ht02@(E21@u + u_pres)

        ux_xi[:, 0] = U_wall_bot*xi[:(N+1),0]
        uy_xi[:, 0] = V_wall_left*xi[::(N+1),0]

        ux_xi[:, N] = U_wall_top*xi[N*(N+1):(N+1)*(N+1), 0]
        uy_xi[:, N] = V_wall_right*xi[N::(N+1), 0]

        # ux_xi[:, 1:N] = np.reshape((u[(N+1):N*(N+1)]+u[:(N-1)*(N+1)])*xi[(N+1):N*(N+1)], ((N+1), (N-1)), order='f')/(2*h[:,None])
        # uy_xi[:, 1:N] = np.reshape((u[N*(N+1):2*N*(N+1)]+u[N*(N+1)-1:2*N*(N+1)-1]), ((N+1), (N)), order='C')[:, 1:]*np.reshape(xi, (N+1, N+1))[:, 1:N]/(2*h[:,None]) 
        ux_xi[:, 1:N] = np.reshape((u[(N+1):N*(N+1)]+u[:(N-1)*(N+1)])*xi[(N+1):N*(N+1)], ((N+1), (N-1)), order='f')/(2*h[:, None])
        uy_xi[:, 1:N] = np.reshape((u[N*(N+1):2*N*(N+1)]+u[N*(N+1)-1:2*N*(N+1)-1]), ((N+1), N), order='C')[:, 1:]*np.reshape(xi, (N+1, N+1))[:, 1:N]/(2*h[:, None])

        convective[:N*(N+1)] = np.reshape(-(uy_xi[:-1]+uy_xi[1:])*h/2, (N*(N+1), 1))
        convective[N*(N+1):2*N*(N+1)] = np.reshape((ux_xi[:-1]+ux_xi[1:])*h/2, (N*(N+1), 1), order='f')
                
        # Set up the right hand side for the equation for the pressure
        VLaplace_xi = VLaplace@xi
                
        f = DIV@( u/dt - convective - VLaplace_xi/Re) + u_norm/dt
        
        # Solve for the pressure
        
        p = LU.solve(f)
        
        # Store the velocity from the previous time level in the vector uold
        
        uold = u
        
        # Update the velocity field
        u = u - dt*(convective + E10@p + VLaplace_xi/Re)
        # Every other 1000 iations check whether you approach steady state and 
        # check whether you satsify conservation of mass. The largest rate at whci 
        # mass is created ot destroyed is denoted my 'maxdiv'. This number should
        # be close to machine precision.
        
        if (step % 100 == 0):
            maxdiv = abs(np.max(DIV@u+u_norm))
            diff = abs(np.max(u-uold))
            
            if print_status:
                print("Step at:", step)
                print("-------")
                print("maxdiv : ",maxdiv)
                print("diff   : ", diff)
                print()

            residual_step_hist.append(step)
            maxdiv_list.append(maxdiv)
            diff_list.append(diff)
        
        step += 1

        #if step % 100==0: print("step at:",step)

    # a = np.array([1, 2, 3, 4])

    # plt.semilogy(residual_step_hist, diff_list)
    # plt.semilogy(residual_step_hist, maxdiv_list)
    # plt.show()


    save_config = {"dt": dt,
                "steps": step,
                "Re": Re,
                "N": N,
                "tol": tol,
                "bcs": {"U_wall_top":  U_wall_top,
                            "U_wall_bot": U_wall_bot,
                            "U_wall_left": U_wall_left,
                            "U_wall_right" : U_wall_right,
                            "V_wall_top" : V_wall_top,
                            "V_wall_bot" : V_wall_bot,
                            "V_wall_left" : V_wall_left,
                            "V_wall_right" : V_wall_right},
                    "residual_steps": residual_step_hist,
                    "diff_list": diff_list,
                    "maxdiv_list" : maxdiv_list,
                    "dual_widths" : h.tolist(),
                    "primal_widths" : th.tolist(),    
                    "h_min" : h_min,
                    "th_min" : th_min,             
                }

    print("Saving data files...")
    np.save(f'data/pressure_N_{N}_Re{Re:.1E}_tol_{tol:.1E}.npy', p)
    np.save(f'data/velocity_N_{N}_Re{Re:.1E}_tol_{tol:.1E}.npy', u)

    with open(f"data/config_N_{N}_Re{Re:.1E}_tol_{tol:.1E}.json", "w") as outfile: 
        json.dump(save_config, outfile)


    print(f"Sucessfully Saved files with prefix data/config_N_{N}_Re{Re:.1E}_tol_{tol:.1E}.EXTENSION")


def main():
    
    import itertools as it

    Re_list = [100,1000]
    N_list = [16, 32, 48, 56, 64]
    tol_list = [1e-3,1e-5,1e-7,1e-9,1e-11,1e-13]

    configs_re_100 = list(it.product([Re_list[0]],N_list,tol_list))
    configs_re_1000 = list(it.product([Re_list[1]],N_list,tol_list))
    
    for Re, N, tol in configs_re_100:
        print(f"Now Running cavity for N = {N}, tol = {tol}, and Re of {Re}")
        
        run_cavity_2d(N, tol, Re)
        print("Finished!")
        print()
    



if __name__ == '__main__':
    main()