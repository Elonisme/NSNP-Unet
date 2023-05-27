# 设置训练集与测试集合的比值为9：1

def random_data(args):
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import transforms
    from dataset import LiverDataset

    x_transforms = transforms.Compose([
        transforms.ToTensor(),  # -> [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # ->[-1,1]
    ])

    # mask只需要转换为tensor
    y_transforms = transforms.ToTensor()

    dataset = LiverDataset(r"./data", transform=x_transforms, target_transform=y_transforms)
    train_size = int(len(dataset) * 0.90)
    test_size = len(dataset) - train_size

    # 导入数据
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloaders = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataloaders = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    train_path = r"./random_data/train_data_" + args.model + ".pth"
    test_path = r"./random_data/test_data_" + args.model + ".pth"
    model_save_path = r"./save_model/" + args.model

    torch.save(train_dataloaders, train_path)
    torch.save(test_dataloaders, test_path)
    return train_path, test_path, model_save_path

