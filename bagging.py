import torch
from mIou import get_iou
from snp_unet import SNP_Unet
from unet import Unet
import numpy as np

Unet_Net_list = [1, 4, 5, 8, 10, 11, 16]
Unet_Nets = 7
SNP_Unet_Net_list = [7, 9, 13, 24, 26]
Snp_Unet_Nets = 5

test_dataloaders = torch.load(r"./random_data/test_data.pt")
len_test = len(test_dataloaders)

device = torch.device("cuda")

net_models = []
mix_model = "yes"
path = ""
if mix_model == "yes":
    for i in range(0, Unet_Nets):
        model = Unet(3, 1).to(device)
        model.load_state_dict(torch.load(r"./save_model/weights-unet-" + str(Unet_Net_list[i]) + ".pth", map_location='cpu'))
        net_models.append(model)

    for i in range(0, Snp_Unet_Nets):
        model = SNP_Unet(3, 1).to(device)
        model.load_state_dict(torch.load(r"./save_model/weights-snp-" + str(SNP_Unet_Net_list[i]) + ".pth", map_location='cpu'))
        net_models.append(model)
    Nets = Unet_Nets+Snp_Unet_Nets
    path = r"save_model/combat-snp-unet.npy"
else:
    for i in range(0, Unet_Nets):
        model = Unet(3, 1).to(device)
        model.load_state_dict(torch.load(r"./save_model/weights-unet-" + str(Unet_Net_list[i]) + ".pth", map_location='cpu'))
        net_models.append(model)
    Nets = Unet_Nets
    path = r"save_model/combat-unet.npy"

n = -1  # 验证集中第i张图

miou_total = 0

combat_iou = []
points = []
for x, y in test_dataloaders:
    combat_img = 0
    n += 1
    x = x.to(device)
    img_y = torch.squeeze(y).cpu().numpy()
    num = len(test_dataloaders)  # 验证集图片的总数
    print('The mount of test dataset is %d' % num)
    with torch.no_grad():
        for j in range(0, Nets):
            print("This is No.%d Net model" % j)
            model = net_models[j]
            model.eval()
            hat_y = model(x)
            img_hat_y = torch.squeeze(hat_y).cpu().numpy()
            one_img_iou = get_iou(img_y, img_hat_y)
            points.append(one_img_iou)
            print('No.%d iou of test img is %f' % (n, one_img_iou))
            combat_img += img_hat_y
        combat_img = combat_img/Nets
        combat_img_iou = get_iou(img_y, combat_img)
        combat_iou.append(combat_img_iou)
        print("combat-iou: %f" % combat_img_iou)
        miou_total += combat_img_iou

Miou_value = miou_total / len_test
print('Miou=%f' % Miou_value)
combat_iou = np.array(combat_iou)
np.save(path, combat_iou)


