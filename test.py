import numpy as np

d_tau = 2*4.4e-6
M = 0.0478
dp = 1e-6
mylambda =  532e-9

diff = np.sqrt(d_tau**2 - (M*dp)**2)

f_stop = diff/(2.44*mylambda*(1+M))

print(f_stop)