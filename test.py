def test(args, a):
    import torch
    from torch.utils.tensorboard import SummaryWriter
    from mIou import get_iou

    model = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "snp_unet":
        from snp_unet import SNP_Unet
        args.ckp = r"./save_model/weights-snp-dilation1-" + str(a) + ".pth"
        model = SNP_Unet(3, 1).to(device)
    elif args.model == "unet":
        from unet import Unet
        args.ckp = r"./save_model/weights-unet-" + str(a) + ".pth"
        model = Unet(3, 1).to(device)

    test_dataloaders = torch.load(r"./random_data/test_data.pt")
    args.ckp = r"./save_model/weights-unet-9.pth"
    writer = SummaryWriter("test_dataset_imgs_logs")

    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        i = 0  # 验证集中第i张图
        miou_total = 0
        num = len(test_dataloaders)  # 验证集图片的总数

        print('The mount of test dataset is %d' % num)
        for x, y in test_dataloaders:
            x = x.to(device)
            hat_y = model(x)

            # 在tensorboard中添加图片
            writer.add_images("test liver", x, i)
            writer.add_images("test tumor", y, i)
            writer.add_images("test predict", hat_y, i)

            img_y = torch.squeeze(y).cpu().numpy()
            img_hat_y = torch.squeeze(hat_y).cpu().numpy()

            print('No.%d iou of test img is %f' % (i, get_iou(img_y, img_hat_y)))
            miou_total += get_iou(img_y, img_hat_y)  # 获取当前预测图的miou，并加到总miou中

            if i < num: i += 1  # 处理验证集下一张图
        print('Miou=%f' % (miou_total / num))
        Miou_value = miou_total / num

        Miou_snp_path = r"./Miou_value/miou-snp-dilation1-" + str(a) + ".pth"
        Miou_unt_path = r"./Miou_value/miou-unet-" + str(a) + ".pth"
        if args.model == "snp_unet":
            torch.save(Miou_value, Miou_snp_path)
        elif args.model == "unet":
            torch.save(Miou_value, Miou_unt_path)

    writer.close()
