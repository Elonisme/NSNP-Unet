import numpy as np
from matplotlib import pyplot as plt

data = np.load("save_model/combat-unet.npy")
plt.bar(range(len(data)), data)
plt.xlabel("test img")
plt.ylabel("Iou")
plt.title("The U-net Ensemble Model's Iou of the different test img")
plt.show()