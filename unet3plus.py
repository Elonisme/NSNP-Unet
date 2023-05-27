import argparse

import torch
import torch.nn as nn
from torch import optim

from random_data import random_data


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


class Unet3plus(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet3plus, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(512, 1024)
        self.conv5out = nn.Conv2d(1024, out_ch, 1)

        self.up6 = nn.ConvTranspose2d(1024, 320, 2, stride=2)
        self.conv6 = DoubleConv(1280, 320)
        self.conv6out = nn.Conv2d(320, out_ch, 1)

        self.up7 = nn.ConvTranspose2d(320, 320, 2, stride=2)
        self.conv7 = DoubleConv(768, 320)
        self.conv7out = nn.Conv2d(320, out_ch, 1)

        self.up8 = nn.ConvTranspose2d(320, 320, 2, stride=2)
        self.conv8 = DoubleConv(512, 320)
        self.conv8out = nn.Conv2d(320, out_ch, 1)

        self.up9 = nn.ConvTranspose2d(320, 320, 2, stride=2)
        self.conv9 = DoubleConv(384, 320)
        self.conv9out = nn.Conv2d(320, out_ch, 1)

        # self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # self.conv6 = DoubleConv(1024, 512)
        # self.conv6out = nn.Conv2d(512, out_ch, 1)
        #
        # self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.conv7 = DoubleConv(512, 256)
        # self.conv7out = nn.Conv2d(256, out_ch, 1)
        #
        # self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # self.conv8 = DoubleConv(256, 128)
        # self.conv8out = nn.Conv2d(128, out_ch, 1)
        #
        # self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # self.conv9 = DoubleConv(128, 64)
        # self.conv9out = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c1_in = nn.functional.interpolate(p1, size=(256, 256), mode='nearest')

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c2_in = nn.functional.interpolate(p2, size=(256, 256), mode='nearest')

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c3_in = nn.functional.interpolate(p3, size=(256, 256), mode='nearest')

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c4_in = nn.functional.interpolate(p4, size=(256, 256), mode='nearest')

        c5 = self.conv5(p4)

        up_6 = self.up6(c5)
        c6_in = nn.functional.interpolate(up_6, size=(256, 256), mode='nearest')
        merge6 = torch.cat([c6_in, c1_in, c2_in, c3_in, c4_in], dim=1)
        c6 = self.conv6(merge6)

        up_7 = self.up7(c6)
        c7_in = nn.functional.interpolate(up_7, size=(256, 256), mode='nearest')
        merge7 = torch.cat([c7_in, c1_in, c2_in, c3_in], dim=1)
        c7 = self.conv7(merge7)

        up_8 = self.up8(c7)
        c8_in = nn.functional.interpolate(up_8, size=(256, 256), mode='nearest')
        merge8 = torch.cat([c8_in, c1_in, c2_in], dim=1)
        c8 = self.conv8(merge8)

        up_9 = self.up9(c8)
        c9_in = nn.functional.interpolate(up_9, size=(256, 256), mode='nearest')
        merge9 = torch.cat([c9_in, c1_in], dim=1)
        c9 = self.conv9(merge9)
        out4 = self.conv9out(c9)
        out4 = nn.functional.interpolate(out4, size=(512, 512), mode='nearest')
        sup4 = nn.Sigmoid()(out4)

        return sup4


if __name__ == "__main__":
    # 是否使用cuda
    parse = argparse.ArgumentParser()
    parse.add_argument("--model", type=str, default="unet3+", help="snp_unet or unet or unet++ or unet3+")
    parse.add_argument("--action", type=str, default="train", help="train or test")
    parse.add_argument("--is_random_data", type=str, default="yes", help="yes or no")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--learn_rate", type=float, default=1e-4)
    parse.add_argument("--num_epochs", type=int, default=30)
    parse.add_argument("--ckp", type=str)
    args = parse.parse_args()

    train_path = ""
    test_path = ""
    model_save_path = ""

    if args.is_random_data == "yes":
        train_path, test_path, model_save_path = random_data(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloaders = torch.load(train_path)

    # 设置模型参数
    model = Unet3plus(3, 1).to(device)
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
        if epoch > 9:
            torch.save(model.state_dict(), model_save_path + str(epoch - 10) + ".pth")

