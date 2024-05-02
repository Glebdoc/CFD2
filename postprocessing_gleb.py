
import matplotlib.pyplot as plt
import numpy as np
import incidence as inc
import hodge as hod
import json


def load_data(N, tol, Re):

    p = np.load(f'data/pressure_N_{N}_Re{Re:.1E}_tol_{tol:.1E}.npy')
    u = np.load(f'data/velocity_N_{N}_Re{Re:.1E}_tol_{tol:.1E}.npy')

    with open(f"data/config_N_{N}_Re{Re:.1E}_tol_{tol:.1E}.json") as json_file:
        config = json.load(json_file)

    return p, u, config


# Re_list = [100,1000]
# N_list = [16, 32, 48, 56, 64]
# tol_list = [1e-3,1e-5,1e-7,1e-9,1e-11,1e-13]

# p, u, config = load_data(64, 1e-13, 100)
def postpr(N, plot_ux, plot_uy, plot_px, plot_py, plot_vort_x, plot_vort_y):

    Re_list = [100,1000]
    N_list = [16, 32, 48, 56, 64]
    tol_list = [1e-3,1e-5,1e-7,1e-9,1e-11,1e-13]

    p, u, config = load_data(N, 1e-13, 1000)

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

    dt = min(h_min,0.5*Re*h_min**2)             # dt for stable integration

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
    
    xi_grid = np.reshape(xi, (N+1, N+1), order='C')

    myXi = E21 @ u + u_pres
    print(myXi.shape)
    myXi = np.reshape(xi,(N+1, N+1))
    print(myXi.shape)
    print(np.max(myXi))

    fluxes = Ht11@u
    hor_fluxes = fluxes[0:int(len(fluxes)/2)]
    ver_fluxes = fluxes[int(len(fluxes)/2):]

    streamFunction = np.zeros([N+1,N+1])
    for j in range(N+1):
        for i in range(N+1):
            if i == 0 and j == 0:
                streamFunction[i,j] = 0
                continue
            streamFunction[i,j] = streamFunction[i-1, j] - ver_fluxes[j*N + i-1]
            if i == 0:
                streamFunction[i,j] = streamFunction[i,j-1] + hor_fluxes[(j-1)*(N+1)]


    # levels_stream = [0.1175, 0.115, 0.11, 0.1, 9e-2, 7e-2, 5e-2, 3e-2, 1e-2, 1e-4, 1e-5, 1e-10, 0, -1e-6, -1e-5, -5e-5,
    #                      -1e-4, -2.5e-4, -5e-4, -1e-3, -1.5e-3]
    # levels_stream.sort()
    # streamFunction = np.flipud(streamFunction)
    # streamFunction = np.rot90(streamFunction, -1)
    # plt.contourf(X_th, Y_th, streamFunction, 
    #             levels=levels_stream, cmap='plasma'
    #             )
    # plt.colorbar()
    # plt.axis("scaled")
    # plt.savefig(f'./plots/stream_func_{N}.pdf')
    # plt.close()

    # levels_vorticity = [-3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4., 5.]
    # levels_vorticity.sort()
    # plt.contourf(X_th, Y_th, xi_grid, levels=levels_vorticity, cmap='plasma')
    # plt.colorbar()
    # plt.axis("scaled")
    # plt.savefig(f'./plots/vorticity_func_{N}.pdf')
    # plt.close()

    # levels_vorticity = [-3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4., 5.]
    # levels_vorticity.sort()
    # plt.pcolormesh(X_th, Y_th, streamFunction)
    # plt.colorbar()
    # plt.axis("scaled")
    # plt.show()

    levels_vorticity = [-3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4., 5.]
    levels_vorticity.sort()
    # plt.colorbar()
    # plt.axis("scaled")
    # plt.show()


    Vu = u[0:len(u)//2,:]
    Vv = u[len(u)//2:,:]
    ux = np.zeros((N,N))
    uy = np.zeros((N,N))
    Vmag = np.zeros((N,N))
    for j in range(N):
        for i in range(N):
            utemp = (Vu[j*(N+1)+i]/h[i] + Vu[j*(N+1)+i+1]/h[i+1])/2
            ux[i,j] = utemp
            vtemp = (Vv[(j+1)*N+i-1]/h[j] + Vv[(j+1)*N+ i]/h[j+1])/2
            uy[i,j] = vtemp
            Vmag[i,j] = np.sqrt(utemp**2 + vtemp**2)

    ux = np.rot90(ux, k=1)
    ux = np.flipud(ux)
    uy = np.rot90(uy, k=1)
    uy = np.flipud(uy)

    Vmag= np.rot90(Vmag, k=1)

    if plot_ux:
        plt.plot(X_th[0,1:], ux[:,N//2], label=f'N={N}')

    if plot_uy:
        plt.plot(Y_th[1:, 0], uy[N//2, :], label=f'N={N}')


    def get_pressure(p, N):
        p_remaining = p[N:len(p)-N]
        p_remaining = p_remaining.reshape(N,N+2)

        p_inner = p_remaining[:, 1:len(p_remaining[0]) - 1]
        return p_inner

    p_inner = get_pressure(p, N)    
    p_inner = np.flipud(p_inner)
    #p_inner = np.rot90(p_inner, k=-1)
    p_static = p_inner - 0.5*1*(Vmag**2) 
    p_static-= p_static[N//2,N//2]

    # plt.imshow(p_inner, cmap='plasma', interpolation='bilinear', vmin=0.05, vmax=0.1)  
    # levels_pressure = [0.3, 0.17, 0.12, 0.11, 0.09, 0.07, 0.05, 0.02, 0.0, -0.002]
    # levels_pressure.sort()
    # print(np.max(p_static), np.min(p_static))    
    # contour = plt.contourf(X_th[1:,1:], Y_th[1:,1:], np.flipud(p_static), levels=levels_pressure, cmap='plasma')
    # plt.colorbar()
    # plt.axis("scaled")
    # plt.savefig(f'./plots/pressure_field{N}.pdf')
    
    p_static = np.flipud(p_static)

    print('myXi.shape', myXi.shape)
    print(np.max(myXi))

    # pressure x profile
    if plot_px:
        plt.plot(Y_th[1:, 0], p_static[N//2,:], label=f'N={N}')
    #pressure y profile
    if plot_py:
        plt.plot(Y_th[1:, 0], p_static[:,N//2], label=f'N={N}')

    if plot_vort_x:
        plt.plot(Y_h[1:, 0], myXi[N//2, :], label=f'N={N}')

    if plot_vort_y:
        plt.plot(Y_h[1:, 0], myXi[:, N//2], label=f'N={N}')


vort_y = [14.7534,12.0670,9.49496,6.95968,4.85754,1.76200,2.09121,2.06539,2.06722,2.06215,2.26772,1.05467,-1.63436,-2.20175,-2.31786,-2.44960,-4.16648]
vort_x = [-5.46217,-8.44350,-8.24616,-7.58524,-6.50867,0.92291,3.43016,2.21171,2.06722,2.06122,2.00174,0.74207,-0.82398,-1.23991,-1.50306,-1.83308,-7.66369]
px = [0.077455,0.078837,0.078685,0.078148,0.077154,0.065816,0.049029,0.034552,0.000000,0.044848,0.047260,0.069511,0.084386,0.086716,0.087653,0.088445,0.090477 ]
py = [0.052987,0.052009,0.051514,0.050949,0.050329,0.034910,0.012122,-0.000827,0.000000,0.004434,0.040377,0.081925,0.104187,0.108566,0.109200,0.109689,0.110591]
y = [1.0000,0.9766,0.9688,0.9609,0.9531,0.8516,0.7344,0.6172,0.5000,0.4531,0.2813,0.1719,0.1016,0.0703,0.0625,0.0547,0.0000]
x = [0.0000,0.0312,0.0391,0.0469,0.0547,0.0937,0.1406,0.1953,0.5000,0.7656,0.7734,0.8437,0.9062,0.9219,0.9297,0.9375,1.0000]
u_exact = [-1.0000000, -0.6644227, -0.5808359, -0.5169277, -0.4723329, -0.3372212, -0.1886747, -0.0570178, 0.0620561, 0.1081999, 0.2803696, 0.3885691, 0.3004561, 0.2228955, 0.2023300, 0.1812881,0.0000000]
v_exact = [0.0000000, -0.2279225, -0.2936869, -0.3553213, -0.4103754,-0.5264392,-0.4264545,-0.3202137,0.0257995,0.3253592,0.3339924,0.3769189,0.3330442,0.3099097,0.2962703,0.2807056,0.0000000]

N_list = [16, 32, 48, 56, 64]
for N in N_list:
    postpr(N, plot_ux=False, plot_uy=False, plot_px=False, plot_py=False, plot_vort_x = False, plot_vort_y = True)
plt.scatter(y, vort_y, color='red', label='Exact')
#plt.scatter(y, py, color='red', label='Exact')
plt.xlabel('y')
plt.ylabel(f'$\omega$')
plt.legend()
plt.grid()
plt.savefig('./plots/vort_y.pdf')