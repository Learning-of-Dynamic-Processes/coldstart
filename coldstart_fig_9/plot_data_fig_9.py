import  matplotlib.pyplot as plt
import numpy as np
# for the scatterplot
try:
    mse = np.loadtxt(f'pred_data/Lorenz_Error_MSE_{9999}.txt')
    err = np.loadtxt(f'pred_data/Lorenz_Error_{9999}.txt')
except:
    raise ValueError('File not found: run the code generate_data_fig_9.py first')
fig = plt.figure()
ax = fig.subplots(1,1)

ax.scatter(err, mse, s=0.2, c='blue', label='MSE')
plt.savefig(f'Img/Lorenz_Error_MSE.png')

