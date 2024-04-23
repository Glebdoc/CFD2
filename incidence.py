import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
N = 5

nVolumes = N*N + N*4
nEdges = 2*(N*N + 3*N)

E21 = sparse.csr_matrix((nVolumes, nEdges), dtype=int)

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

# a = E21.todense()
# plt.imshow(a, cmap='viridis', interpolation='nearest')
# plt.title('Visualization of Sparse Matrix')
# plt.colorbar()
# plt.show()

#create a list of edges that needs to be removed from the sparse matrix
Ni = np.arange(N)
LB = Ni*(N+3)
RB = Ni*(N+3) + N+2
BB = np.arange(int(nEdges/2), int(nEdges/2) + N)
UB = np.arange(nEdges - N, nEdges)

columns_to_remove = np.concatenate((LB, RB, BB, UB))
columns_to_remove = np.sort(columns_to_remove)

all_columns = np.arange(E21.shape[1])
columns_to_keep = np.setdiff1d(all_columns, columns_to_remove)

E21_modified = E21[:, columns_to_keep]
E_extra = sparse.csr_matrix((nVolumes, len(columns_to_remove)), dtype=int)
a = E21_modified.todense()

for idx, col_idx in enumerate(columns_to_remove):
    column_to_move = E21[:, col_idx].toarray()
    E_extra[:, idx] = column_to_move


b = E_extra.todense()

fig, axs = plt.subplots(1, 2)

axs[0].imshow(a, cmap='viridis', interpolation='nearest')
axs[0].set_title('Visualization of Sparse Matrix (a)')
axs[0].axis('off')

axs[1].imshow(b, cmap='viridis', interpolation='nearest')
axs[1].set_title('Visualization of Sparse Matrix (b)')
axs[1].axis('off')

plt.show()