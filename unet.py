import argparse

import torch
import torch.nn as nn
from torch import optim
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            # nn.BatchNorm2d(in_ch),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, 3, padding=1),

            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)

        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)

        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)

        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

if __name__ == "__main__":
    # 是否使用cuda
    parse = argparse.ArgumentParser()
    parse.add_argument("--model", type=str, default="unet++", help="snp_unet or unet or unet++")
    parse.add_argument("--action", type=str, default="train", help="train or test")
    parse.add_argument("--is_random_data", type=str, default="no", help="yes or no")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--learn_rate", type=float, default=1e-3)
    parse.add_argument("--num_epochs", type=int, default=20)
    parse.add_argument("--ckp", type=str)
    args = parse.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ath = r"./save_model/weights-unet-"

    # 设置tensorboard的文件路径

    train_dataloaders = torch.load(r"random_data/train_data.pt")

    # 设置模型参数
    model = Unet(3, 1).to(device)
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
            train_step = train_step + 1
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        break
        # if epoch > 9:
        #     torch.save(model.state_dict(), path + str(epoch - 10) + ".pth")
    # torch.save(model.state_dict(), path+".pth")