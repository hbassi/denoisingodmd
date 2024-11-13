import numpy as np 
import utils as ut
from scipy.linalg import svd, hankel, eig
from matplotlib import pyplot as plt
import scipy
from tqdm import trange
from matplotlib import pyplot as plt

Tmax = 750
t = np.arange(20, Tmax, 1)
sinusoid = np.exp(1j* -2 * t) + 0.01*np.random.randn(len(t))

dataS = sinusoid.real
dt = 1
tol = [1e-2, 1e-3, 1e-4]

lamt,t = ut.run_compare(dataS,dt,tol=tol, Tmax=Tmax,step=1)
#print(lamt,t)
ut.plot_compare(
    t=t,
    lam=lamt,
    tol=tol,
    E=[-2],  
    mytitle="ODMD Frequency Estimation for a Single Sinusoid",
    xlimits=(20, Tmax),
    ylimits=(1e-14, 1e2)
)