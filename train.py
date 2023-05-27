
def train(args):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from torch import optim

    # 是否使用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = ""
    model = ""
    if args.model == "snp_unet":
        from snp_unet import SNP_Unet
        path = r"./save_model/weights-snp-dilation1-"
        model = SNP_Unet(3, 1).to(device)
    elif args.model == "unet":
        from unet import Unet
        path = r"./save_model/weights-unet-"
        model = Unet(3, 1).to(device)
    elif args.model == "unet++":
        from unetpp import UnetPlusPlus
        path = r"./save_model/weights-unetpp-"

    # 设置tensorboard的文件路径
    writer = SummaryWriter("dataset_imgs_logs")
    train_dataloaders = torch.load(r"random_data/train_data.pt")

    # 设置模型参数

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
            writer.add_scalar("train_loss", loss.item(), train_step)
            train_step = train_step + 1
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        if epoch > 9:
            torch.save(model.state_dict(), path+str(epoch-10)+".pth")
    # torch.save(model.state_dict(), path+".pth")

    writer.close()