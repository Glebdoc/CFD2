from scipy.sparse import linalg as splinalg
import scipy.sparse as sparse
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


