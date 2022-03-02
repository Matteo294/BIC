from os import read
from pandas import read_csv
from matplotlib import pyplot as plt 
import numpy as np

gauss = read_csv("gauss.csv")
gauss100 = read_csv("gauss100.csv")
gauss1000 = read_csv("gauss1000.csv")
jacobi = read_csv("jacobi.csv")
jacobi100 = read_csv("jacobi100.csv")

plt.plot(gauss['x'], gauss['sol'], color='red', lw=1.8, label="8 points")
plt.plot(gauss100['x'], gauss100['sol'], color='mediumspringgreen', lw=1.8, label="100 points")
plt.plot(gauss1000['x'], gauss1000['sol'], color='blue', lw=1.8, label="1000 points")
plt.xlabel("x", fontsize=14)
plt.ylabel("T(x)", fontsize=14)
plt.legend()
plt.savefig("gauss.eps", dpi=300)
plt.show()

plt.plot(gauss1000['x'], gauss1000['sol'], label='FEBS', color='black', lw=2.0)
for n,row in enumerate(jacobi.to_numpy()):
    if (n+1) % 10 == 0:
        plt.plot(gauss['x'], row, label="Iteration "+str(n+1))
plt.legend()
plt.xlabel("x", fontsize=14)
plt.ylabel("T(x)", fontsize=14)
plt.savefig("jacobi8.eps", dpi=300)
plt.show()

plt.plot(gauss1000['x'], gauss1000['sol'], label='FEBS', color='black', lw=2.0)
for n,row in enumerate(jacobi100.to_numpy()):
    if (n+1) % 10 == 0:
        plt.plot(gauss100['x'], row, label="Iteration "+str(n+1))
plt.legend()
plt.xlabel("x", fontsize=14)
plt.ylabel("T(x)", fontsize=14)
plt.savefig("jacobi100.eps", dpi=300)
plt.show()
