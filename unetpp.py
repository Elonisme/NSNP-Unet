import torch
import torch.nn as nn
from torch import optim
import argparse

from mIou import get_iou


class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.Sigmoid(),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))

        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.Sigmoid(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.Sigmoid())

    def forward(self, x):
        x = self.Conv_forward(x)
        return x


class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes, deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]

        self.CONV3_1 = ContinusParalleConv(512 * 2, 512, pre_Batch_Norm=True)

        self.CONV2_2 = ContinusParalleConv(256 * 3, 256, pre_Batch_Norm=True)
        self.CONV2_1 = ContinusParalleConv(256 * 2, 256, pre_Batch_Norm=True)

        self.CONV1_1 = ContinusParalleConv(128 * 2, 128, pre_Batch_Norm=True)
        self.CONV1_2 = ContinusParalleConv(128 * 3, 128, pre_Batch_Norm=True)
        self.CONV1_3 = ContinusParalleConv(128 * 4, 128, pre_Batch_Norm=True)

        self.CONV0_1 = ContinusParalleConv(64 * 2, 64, pre_Batch_Norm=True)
        self.CONV0_2 = ContinusParalleConv(64 * 3, 64, pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(64 * 4, 64, pre_Batch_Norm=True)
        self.CONV0_4 = ContinusParalleConv(64 * 5, 64, pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(3, 64, pre_Batch_Norm=False)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm=False)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm=False)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm=False)

        self.pool = nn.MaxPool2d(2)

        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))

        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], 1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)

        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)

        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)

        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)

        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)


if __name__ == "__main__":
    print("deep_supervision: False")
    deep_supervision = False
    device = torch.device('cuda')
    # inputs = torch.randn((1, 3, 512, 512)).to(device)
    # model = UnetPlusPlus(num_classes=1, deep_supervision=deep_supervision).to(device)
    # outputs = model(inputs)
    # print(outputs.shape)
    #
    # print("deep_supervision: True")
    # deep_supervision = True
    # model = UnetPlusPlus(num_classes=3, deep_supervision=deep_supervision).to(device)
    # outputs = model(inputs)
    # for out in outputs:
    #     print(out.shape)

    parse = argparse.ArgumentParser()
    parse.add_argument("--model", type=str, default="unet++", help="snp_unet or unet or unet++")
    parse.add_argument("--action", type=str, default="train", help="train or test")
    parse.add_argument("--is_random_data", type=str, default="no", help="yes or no")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--learn_rate", type=float, default=1e-5)
    parse.add_argument("--num_epochs", type=int, default=20)
    parse.add_argument("--ckp", type=str)
    args = parse.parse_args()
    path = r"./save_model/weights-unetpp-"

    train_dataloaders = torch.load(r"random_data/train_data1.pt")

    # 设置模型参数
    model = UnetPlusPlus(1, deep_supervision=deep_supervision).to(device)
    learn_rate = args.learn_rate
    num_epochs = args.num_epochs
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # 训练模型
    train_step = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloaders.dataset)
        epoch_loss = 0
        step = 0
        for x, y in train_dataloaders:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_dataloaders.batch_size + 1, loss.item()))
            # writer.add_scalar("train_loss", loss.item(), train_step)
            train_step = train_step + 1
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        if epoch > 9:
            torch.save(model.state_dict(), path + str(epoch - 10) + ".pth")

    # test
    args.ckp = r"./save_model/weights-unetpp-10.pth"
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    model.eval()
    test_dataloaders = torch.load(r"./random_data/test_data.pt")
    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        num = len(test_dataloaders)  # 验证集图片的总数

        print('The mount of test dataset is %d' % num)
        for x, y in test_dataloaders:
            x = x.to(device)
            hat_y = model(x)

            # 在tensorboard中添加图片
            img_y = torch.squeeze(y).cpu().numpy()
            img_hat_y = torch.squeeze(hat_y).cpu().numpy()

            print('No.%d iou of test img is %f' % (i, get_iou(img_y, img_hat_y)))
            miou_total += get_iou(img_y, img_hat_y)  # 获取当前预测图的miou，并加到总miou中

            if i < num: i += 1  # 处理验证集下一张图
        print('Miou=%f' % (miou_total / num))
        Miou_value = miou_total / num


        Miou_untpp_path = r"./Miou_value/miou-unetpp.pth"
        torch.save(Miou_value, Miou_untpp_path)

    # torch.save(model.state_dict(), path+".pth")
