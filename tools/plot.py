import os
import sys
import numpy as np
from matplotlib import pyplot as plt

base_dir = sys.argv[1]

total_mAP = np.load(os.path.join(base_dir, "total_mAP.npy"))
total_mAP50 = np.load(os.path.join(base_dir, "total_mAP50.npy"))
plt.figure()
plt.plot(total_mAP.reshape(-1), label="mAP50:95")
plt.plot(total_mAP50.reshape(-1), label="mAP50")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("mAP")
plt.savefig("./mAP.jpg")
# plt.show()

total_heatmap_losses = np.load(os.path.join(base_dir, "total_heatmap_losses.npy"))
total_wh_losses = np.load(os.path.join(base_dir, "total_wh_losses.npy"))
total_kd_losses = np.load(os.path.join(base_dir, "total_kd_losses.npy"))
total_losses = np.load(os.path.join(base_dir, "total_losses.npy"))
plt.figure()
plt.plot(total_heatmap_losses.reshape(-1), label="cls")
plt.plot(total_wh_losses.reshape(-1), label="wh")
plt.plot(total_kd_losses.reshape(-1), label="kd")
plt.plot(total_losses.reshape(-1), label="total")
plt.legend()
plt.xlabel("iter")
plt.ylabel("loss")
plt.savefig("./loss.jpg")
# plt.show()
