import torch
import matplotlib.pyplot as plt

L1 = []
for i in range(0, 40):
    L1.append(torch.load("./Miou_value/miou-snp-"+str(i)+".pth"))

L2 = range(10, 50)

plt.plot(L2, L1, 's-', color='r', label="ATT-RLSTM")
plt.xlabel("epoch")
plt.ylabel("Miou")
plt.title("The SNP-U-net miou of the diffrent Epoch")
plt.show()