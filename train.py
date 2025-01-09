from torch.utils.tensorboard import SummaryWriter
from pix2Topix import pix2pixG_256, pix2pixD_256
import argparse
from mydatasets import CreateDatasets
from split_data import split_data
import os
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim as optim
import torch.nn as nn
from utils import train_one_epoch, val


def train(opt):
    batch = opt.batch
    data_path = opt.dataPath
    print_every = opt.every
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = opt.epoch
    img_size = opt.imgsize

    if not os.path.exists(opt.savePath):
        os.mkdir(opt.savePath)

    # 加载数据集
    train_imglist, val_imglist = split_data(data_path)
    train_datasets = CreateDatasets(train_imglist, img_size)
    val_datasets = CreateDatasets(val_imglist, img_size)

    train_loader = DataLoader(dataset=train_datasets, batch_size=batch, shuffle=True, num_workers=opt.numworker,
                              drop_last=True)
    val_loader = DataLoader(dataset=val_datasets, batch_size=batch, shuffle=True, num_workers=opt.numworker,
                            drop_last=True)

    # 实例化网络
    pix_G = pix2pixG_256().to(device)
    pix_D = pix2pixD_256().to(device)

    # 定义优化器和损失函数
    optim_G = optim.Adam(pix_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_D = optim.Adam(pix_D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    start_epoch = 0

    # 加载预训练权重
    if opt.weight != '':
        ckpt = torch.load(opt.weight)
        pix_G.load_state_dict(ckpt['G_model'], strict=False)
        pix_D.load_state_dict(ckpt['D_model'], strict=False)
        start_epoch = ckpt['epoch'] + 1

    writer = SummaryWriter('train_logs')
    # 开始训练
    for epoch in range(start_epoch, epochs):
        loss_mG, loss_mD = train_one_epoch(G=pix_G, D=pix_D, train_loader=train_loader,
                                           optim_G=optim_G, optim_D=optim_D, writer=writer, loss=loss, device=device,
                                           plot_every=print_every, epoch=epoch, l1_loss=l1_loss)

        writer.add_scalars(main_tag='train_loss', tag_scalar_dict={
            'loss_G': loss_mG,
            'loss_D': loss_mD
        }, global_step=epoch)

        # 验证集
        val(G=pix_G, D=pix_D, val_loader=val_loader, loss=loss, l1_loss=l1_loss, device=device, epoch=epoch)
        # 保存模型
        torch.save({
            'G_model': pix_G.state_dict(),
            'D_model': pix_D.state_dict(),
            'epoch': epoch
        }, './weights/pix2pix_256.pth')


def cfg():
    parse = argparse.ArgumentParser()
    parse.add_argument('--batch', type=int, default=16)
    parse.add_argument('--epoch', type=int, default=200)
    parse.add_argument('--imgsize', type=int, default=256)
    parse.add_argument('--dataPath', type=str, default='../base', help='data root path')
    parse.add_argument('--weight', type=str, default='weights/pix2pix_256.pth', help='load pre train weight')
    parse.add_argument('--savePath', type=str, default='./weights', help='weight save path')
    parse.add_argument('--numworker', type=int, default=4)
    parse.add_argument('--every', type=int, default=2, help='plot train result every * iters')
    opt = parse.parse_args()
    return opt


if __name__ == '__main__':
    opt = cfg()
    print(opt)
    train(opt)
