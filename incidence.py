import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
N = 3

def move(matrix, columns_to_remove, nVolumes):
    all_columns = np.arange(matrix.shape[1])
    columns_to_keep = np.setdiff1d(all_columns, columns_to_remove)

    matrix_modified = matrix[:, columns_to_keep]

    matrix_extra = sparse.lil_matrix((nVolumes, len(columns_to_remove)), dtype=int)

    for idx, col_idx in enumerate(columns_to_remove):
        column_to_move = matrix[:, col_idx].toarray()
        matrix_extra[:, idx] = column_to_move

    return matrix_modified, matrix_extra

def computetE21(N):

    nVolumes = N*N + N*4
    nEdges = 2*(N*N + 3*N)

    E21 = sparse.lil_matrix((nVolumes, nEdges), dtype=int)

    for j in range(N+2):
        for i in range(N+2):
            if i == 0 and j == 0:
                continue
            if i == 0 and j == N+1:
                continue 
            if i == N+1 and j == 0:
                continue
            if i == N+1 and j == N+1:
                continue
            # # Bottom boundaries
            if j == 0:
                E21[i-1, int(nEdges/2 + i)-1] = -1
                E21[i-1, int(nEdges/2 + i + N)-1] = 1
                continue
            # Left boundaries
            if i == 0:
                E21[N + (j-1)*N + (j-1)*2, (j-1)*N + (j-1)*2 + j-1] = -1
                E21[N + (j-1)*N + (j-1)*2, (j-1)*N + (j-1)*2 + j] = 1
                continue
            
            # Right boundaries FIX RIGHT BOUNDARY
            if i == N+1:
                E21[N + j*N + (j-1)*2 + 1, N + (j-1)*N + (j-1)*2 + j] = -1
                E21[N + j*N + (j-1)*2 + 1, N + (j-1)*N + (j-1)*2 + j+1] = 1
                continue
            
            # Upper boundaries 
            if j == N+1:
                E21[-N + i-1, -2*N + i-1] = -1
                E21[-N + i-1, -N + i-1] = 1
                continue
            
            # Normal seq
            E21[N*j + 2*(j-1) + i, (j-1)*N + (j-1)*3 +i] = -1
            E21[N*j + 2*(j-1) + i, (j-1)*N + (j-1)*3 +i +1] = 1
            E21[N*j + 2*(j-1) + i, int(nEdges/2 + N*j+i-1)] = -1
            E21[N*j + 2*(j-1) + i, int(nEdges/2 + N*j + N +i-1)] = 1

    #create a list of edges that needs to be removed from the sparse matrix
    Ni = np.arange(N)
    LB = Ni*(N+3)
    RB = Ni*(N+3) + N+2
    BB = np.arange(int(nEdges/2), int(nEdges/2) + N)
    UB = np.arange(nEdges - N, nEdges)

    columns_to_remove = np.concatenate((LB, RB, BB, UB))
    #columns_to_remove = np.sort(columns_to_remove)
    matrix, matrix_extra = move(E21, columns_to_remove, nVolumes)
    return matrix.tocsc(), matrix_extra.tocsc()



# compute E21
def compute_dual_E21(N):
    nSurfaces = (N+1)**2
    nEdges = 2*nSurfaces + 2*(N+1)

    E21 = sparse.lil_matrix((nSurfaces, nEdges), dtype=int)
    E21.setdiag(1, k=0)
    E21.setdiag(-1, k=N+1)

    for j in range(N+1):
        for i in range(N+1):
            E21[j*(N+1) + i, int(nEdges/2) + j*(N+1) + i + j] = -1
            E21[j*(N+1) + i, int(nEdges/2) + j*(N+1) + i + j +1] = 1

    # Now lets remove columns that correspond to the boundaries
    Ni = np.arange(N+1)
    BB = np.arange(N+1)
    UB = np.arange(int(nEdges/2) - N-1, int(nEdges/2))
    RB = int(nEdges/2) + Ni*(N+2)
    LB = int(nEdges/2) + Ni*(N+2) + N+1

    columns_to_remove = np.concatenate((LB, RB, BB, UB))
    #columns_to_remove = np.sort(columns_to_remove)

    matrix, matrix_extra = move(E21, columns_to_remove, nSurfaces)
    return matrix.tocsc(), matrix_extra.tocsc()

tE21, tE21_extra = computetE21(N)
E21, E21_extra = compute_dual_E21(N)

E10 = -tE21.transpose()

def plotMatrix(matrix):
    num_rows, num_cols = matrix.shape
    plt.xticks(np.arange(num_cols)-0.5, np.arange(num_cols))  # Shift x-ticks by -0.5
    plt.yticks(np.arange(num_rows)-0.5, np.arange(num_rows))  # Shift y-ticks by -0.5
    plt.grid(True, which='both', linestyle='-', color='k', linewidth=1)  # Add gridlines
    plt.imshow(matrix.todense(), cmap='plasma', interpolation='nearest')
    plt.colorbar()
    plt.show()

plotMatrix(E10)
# tE21, tE21_extra = computetE21(N) 
# print(tE21.shape)
# E21, E21_extra = compute_dual_E21(N)
# print(E21.shape)











# fig, axs = plt.subplots(1, 2)

# axs[0].imshow(E21.todense(), cmap='binary', interpolation='nearest')
# axs[0].set_title('Visualization of Sparse Matrix (a)')
# axs[0].grid(True, linestyle='--', linewidth=0.5, color='gray')  # Customize grid appearance
# axs[0].axis('off')

# axs[1].imshow(E21_extra.todense(), cmap='binary', interpolation='nearest')
# axs[1].set_title('Visualization of Sparse Matrix (b)')
# axs[1].axis('off')
# num_rows, num_cols = E21_extra.shape
# plt.imshow(E21_extra.todense(), cmap='binary', interpolation='nearest', )
# plt.grid(True, which='both', linestyle='-', color='k', linewidth=1)  # Add gridlines

# plt.xticks(np.arange(num_cols)-0.5, np.arange(num_cols))  # Shift x-ticks by -0.5
# plt.yticks(np.arange(num_rows)-0.5, np.arange(num_rows))  # Shift y-ticks by -0.5

# plt.colorbar()
# plt.show()

