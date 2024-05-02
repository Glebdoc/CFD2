
import matplotlib.pyplot as plt
import numpy as np
import incidence as inc
import hodge as hod
import json
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts}'
})


def load_data(N, tol, Re):

    p = np.load(f'data/pressure_N_{N}_Re{Re:.1E}_tol_{tol:.1E}.npy')
    u = np.load(f'data/velocity_N_{N}_Re{Re:.1E}_tol_{tol:.1E}.npy')

    with open(f"data/config_N_{N}_Re{Re:.1E}_tol_{tol:.1E}.json") as json_file:
        config = json.load(json_file)

    return p, u, config

def get_pressure(p, N):
    p_remaining = p[N:len(p)-N]
    p_remaining = p_remaining.reshape(N,N+2)

    p_inner = p_remaining[:, 1:len(p_remaining[0]) - 1]
    return p_inner

y_th_list=[]
p_list =[]


Re_list = [100,1000]
N_list = [16, 32, 48, 56, 64]
tol_list = [1e-3,1e-5,1e-7,1e-9,1e-11,1e-13]


cent_vort=[]

for N in N_list:
    p, u, config = load_data(N, 1e-13, 100)

    tol = config['tol']

    L = 1
    Re = config['Re']    # Reynolds number 
    N = config['N']   		# mesh cells in x- and y-direction

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

    dt = config['dt']             # dt for stable integration

    U_wall_top = config['bcs']['U_wall_top']
    U_wall_bot = config['bcs']['U_wall_bot']
    U_wall_left = config['bcs']['U_wall_left']
    U_wall_right = config['bcs']['U_wall_right']
    V_wall_top = config['bcs']['V_wall_top']
    V_wall_bot = config['bcs']['V_wall_bot']
    V_wall_left = config['bcs']['V_wall_left']
    V_wall_right = config['bcs']['V_wall_right']


    # Set up the sparse incidence matrix tE21. Use the orientations described
    # in the assignment.
    # Make sure to sue sparse matrices to avoid memory problems

    tE21, tE21_norm = inc.computetE21(N)
    E10 = -tE21.transpose()
    E21, E21_norm = inc.compute_dual_E21(N)
    tE10 = E21.transpose()

    # Setup u_norm
    LB = U_wall_left*np.ones(N) * th 
    RB = U_wall_right*np.ones(N) * th
    TB = V_wall_top*np.ones(N) * th
    BB = V_wall_bot*np.ones(N) * th
    u_norm = np.concatenate((LB, RB, BB, TB), axis=0)
    # Insert the normal boundary conditions and split of the vector u_norm
    u_norm = tE21_norm @ u_norm
    u_norm = u_norm[:,np.newaxis]


    #  Split off 
    # the prescribed tangential velocity and store this in 
    #  the vector u_pres
    LB = V_wall_left*np.ones(N+1) * h 
    RB = V_wall_right*np.ones(N+1) * h
    BB = U_wall_bot*np.ones(N+1) * h
    TB = U_wall_top*np.ones(N+1) * h
    u_pres = np.concatenate((LB, RB, BB, TB), axis=0)
    u_pres = (E21_norm @ u_pres)[:,np.newaxis]

    #  Set up the Hodge matrices Ht11 and H1t1
    H1t1, Ht11= hod.get_Ht11(th, h, N)
    Ht02, H2t0, _  = hod.get_Ht02(h, N)

    residual_step_hist = config['residual_steps']
    diff_list = config['diff_list']
    maxdiv_list = config['maxdiv_list']

    coord_stack_th = np.cumsum(np.insert(th,0,0))
    X_th, Y_th = np.meshgrid(coord_stack_th,coord_stack_th, indexing='xy')
    coord_stack_h = np.cumsum(np.insert(h,0,0))
    X_h, Y_h = np.meshgrid(coord_stack_h,coord_stack_h, indexing='xy')
    u_pres_vort = Ht02 @ u_pres
    xi = (Ht02 @ E21 @ u + u_pres_vort).flatten()

    myXi = E21 @ u + u_pres
    print("integrated vorticity:", np.sum(myXi))
    print(myXi.shape)
    myXi = np.reshape(xi,(N+1, N+1))
    print(myXi.shape)
    print(np.max(myXi))

    print('myXi.shape', myXi.shape)
    print(np.max(myXi))

    u_pres_vort = Ht02 @ u_pres
    xi = (Ht02 @ E21 @ u + u_pres_vort).flatten()
    xi_grid = np.reshape(xi, (N+1, N+1), order='C')
    xi = (Ht02 @ E21 @ u + u_pres_vort).flatten()
    
    xi_grid = np.reshape(xi, (N+1, N+1), order='C')

    myXi = E21 @ u + u_pres
    myXi = np.reshape(xi,(N+1, N+1))
    # fluxes = Ht11@u
    # hor_fluxes = fluxes[0:int(len(fluxes)/2)]
    # ver_fluxes = fluxes[int(len(fluxes)/2):]

    # streamFunction = np.zeros([N+1,N+1])
    # for j in range(N+1):
    #     for i in range(N+1):
    #         if i == 0 and j == 0:
    #             streamFunction[i,j] = 0
    #             continue
    #         streamFunction[i,j] = streamFunction[i-1, j] - ver_fluxes[j*N + i-1]
    #         if i == 0:
    #             streamFunction[i,j] = streamFunction[i,j-1] + hor_fluxes[(j-1)*(N+1)]


    levels_stream = [0.1175, 0.115, 0.11, 0.1, 9e-2, 7e-2, 5e-2, 3e-2, 1e-2, 1e-4, 1e-5, 1e-10, 0, -1e-6, -1e-5, -5e-5,
                        -1e-4, -2.5e-4, -5e-4, -1e-3, -1.5e-3]
    levels_stream.sort()
    # streamFunction = np.flipud(streamFunction)
    # streamFunction = np.rot90(streamFunction, -1)
    # plt.contourf(X_th, Y_th, streamFunction, 
    #             levels=levels_stream, cmap='plasma'
    #             )
    # plt.colorbar()
    # plt.axis("scaled")
    # plt.show()

    # levels_vorticity = [-3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4., 5.]
    # levels_vorticity.sort()
    # plt.contourf(X_th, Y_th, xi_grid, levels=levels_vorticity, cmap='plasma')
    # plt.colorbar()
    # plt.axis("scaled")
    # plt.show()


    # levels_vorticity = [-3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4., 5.]
    # levels_vorticity.sort()
    # plt.pcolormesh(X_th, Y_th, streamFunction)
    # plt.colorbar()
    # plt.axis("scaled")
    # plt.show()

    levels_vorticity = [-3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4., 5.]
    levels_vorticity.sort()
    print(f"centre value at {xi_grid.shape[0]//2, xi_grid.shape[1]//2} :",xi_grid[xi_grid.shape[0]//2,xi_grid.shape[1]//2])
    print(coord_stack_th[th.size//2],coord_stack_th[th.size//2])
    cent_vort.append(xi_grid[xi_grid.shape[0]//2,xi_grid.shape[1]//2])
    # plt.colorbar()
    # plt.axis("scaled")
    # plt.show()

    # print("p", p.size)


    # print("Vorticity shape", xi.shape)


    # hor_fluxes = fluxes[0:int(len(fluxes)/2)]
    # ver_fluxes = fluxes[int(len(fluxes)/2):]

    # vel_mag = np.zeros([N,N])
    # for j in range(N):
    #     for i in range(N):
    #         if i == 0 and j == 0:
    #             vel_mag[i,j] = 0
    #             continue
    #         vel_mag[i,j] = (hor_fluxes[i-1, j]**2 + ver_fluxes[j*N + i-1]**2)**0.5
    #         if i == 0:
    #             vel_mag[i,j] = streamFunction[i,j-1] + hor_fluxes[(j-1)*(N+1)]

    # p_inner = get_pressure(p, N)
    # print("Pressure inner:", p_inner.shape)
    # plt.imshow(p_inner)
    # plt.colorbar()
    # plt.show()

    # print(u.size)



    # Vu = u[0:len(u)//2,:]
    # Vv = u[len(u)//2:,:]
    # ux = np.zeros((N,N))
    # uy = np.zeros((N,N))
    # Vmag = np.zeros((N,N))
    # for j in range(N):
    #     for i in range(N):
    #         utemp = (Vu[j*(N+1)+i]/h[i] + Vu[j*(N+1)+i+1]/h[i+1])/2
    #         ux[i,j] = utemp
    #         vtemp = (Vv[(j+1)*N+i-1]/h[j] + Vv[(j+1)*N+ i]/h[j+1])/2
    #         uy[i,j] = vtemp
    #         Vmag[i,j] = np.sqrt(utemp**2 + vtemp**2)

    # ux = np.rot90(ux, k=1)
    # ux = np.flipud(ux)
    # uy = np.rot90(uy, k=1)
    # uy = np.flipud(uy)

    # Vmag= np.rot90(Vmag, k=1)



    # # if plot_ux:
    # #     plt.plot(X_th[0,1:], ux[:,N//2])

    # # if plot_uy:
    # #     plt.plot(Y_th[1:, 0], uy[:,N//2-1], label=f'N={N}')

    # p_inner = get_pressure(p, N)    
    # p_inner = np.flipud(p_inner)
    # #p_inner = np.rot90(p_inner, k=-1)
    # p_static = p_inner - 0.5*1*(Vmag**2) 
    # p_static-= p_static[N//2,N//2]

    # # plt.imshow(p_inner, cmap='plasma', interpolation='bilinear', vmin=0.05, vmax=0.1)  
    # # levels_pressure = [0.3, 0.17, 0.12, 0.11, 0.09, 0.07, 0.05, 0.02, 0.0, -0.002]
    # # levels_pressure.sort()
    # # print(np.max(p_static), np.min(p_static))    
    # # contour = plt.contourf(X_th[1:,1:], Y_th[1:,1:], np.flipud(p_static), levels=levels_pressure, cmap='plasma')
    # # plt.colorbar()
    # # plt.axis("scaled")
    # # plt.savefig(f'./plots/pressure_field{N}.pdf')

    # p_static = np.flipud(p_static)

    # # pressure x profile
    # # if plot_px:
    # #     plt.plot(Y_th[1:, 0], p_static[N//2,:], label=f'N={N}')
    # #pressure y profile

    # # if plot_py:
    # #     plt.plot(Y_th[1:, 0], p_static[:,N//2], label=f'N={N}')

    # y_th_list.append(Y_th[1:, 0])
    # p_list.append( p_static[N//2,:])


# for tol,y_th,p in zip(tol_list,y_th_list,p_list):

#     plt.plot(y_th,p,label=f"{tol:.1E}")
    
plt.plot(N_list,cent_vort,label="Numerical Result")
plt.plot(N_list,[1.174412]*len(N_list),'r-',label="Botella Peyret")
plt.xlabel(r"$N$")
plt.ylabel(r"Vorticity")
plt.legend()
plt.grid()
plt.show()



