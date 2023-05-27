import argparse
from random_data import random_data
from test import test
from train import train



# 参数解析器,用来解析从终端读取的命令
parse = argparse.ArgumentParser()
parse.add_argument("--model", type=str, default="unet", help="snp_unet or unet or unet++")
parse.add_argument("--action", type=str, default="test", help="train or test")
parse.add_argument("--is_random_data", type=str, default="no", help="yes or no")
parse.add_argument("--batch_size", type=int, default=1)
parse.add_argument("--learn_rate", type=float, default=1e-5)
parse.add_argument("--num_epochs", type=int, default=50)
parse.add_argument("--ckp", type=str)
args = parse.parse_args()

if args.is_random_data == "yes":
    random_data(args)

if args.action == "train":
    train(args)
    for i in range(0, 20):
        test(args, i)

if args.action == "test":
    test(args, 9)
    # for i in range(0, 40):
    #     test(args, i)

