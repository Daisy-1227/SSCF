import matplotlib.pyplot as plt
import numpy as np
def vis_loss(loss_all):
    loss_all = np.array(loss_all)
    for i in range(loss_all.shape[1]):
        plt.subplot(1,loss_all.shape[1],i+1)
        plt.plot(np.arange(loss_all.shape[0]),loss_all[:,i])
    plt.show()