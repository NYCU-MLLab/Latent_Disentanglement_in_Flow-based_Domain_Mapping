from tqdm import tqdm
import numpy as np
from PIL import Image
from math import log, sqrt, pi

import argparse
import os
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from torch.nn.utils import clip_grad_norm_

from model import Glow


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Glow trainer')
parser.add_argument('--batch', default=10, type=int, help='batch size')
parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
parser.add_argument(
    '--n_flow', default=32, type=int, help='number of flows in each block'
)
parser.add_argument('--n_block', default=4, type=int, help='number of blocks')
parser.add_argument(
    '--no_lu',
    action='store_true',
    help='use plain convolution instead of LU decomposed version',
)
parser.add_argument(
    '--affine', action='store_true', help='use affine coupling instead of additive'
)
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=10, type=int, help='number of samples')
parser.add_argument('-n','--name', type=str, help='Name for path')
parser.add_argument('path', metavar='PATH', type=str, help='Path to image directory')


def sample_data(path, batch_size, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ]
    )

    dataset = datasets.ImageFolder(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)


def calc_z_shapes(n_channel, input_size, n_flow, n_block):
    z_shapes = []

    for i in range(n_block - 1):
        input_size //= 2
        n_channel *= 2

        z_shapes.append((n_channel, input_size, input_size))

    input_size //= 2
    z_shapes.append((n_channel * 4, input_size, input_size))

    return z_shapes


def calc_loss(log_p, logdet, log_c, beta, image_size, n_bins):
    # log_p = calc_log_p([z_list])
    n_pixel = image_size * image_size * 3

    loss = -log(n_bins) * n_pixel
    loss = loss + logdet + log_p + (beta+1)*log_c

    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
        (log_c / (log(2) * n_pixel)).mean()
    )

def Mask(input):
    mask_input = input
    mask_input[:,:,31:47,21:42] = torch.randn_like(input[:,:,31:47,21:42])
    # l26 r37 u36 d42
    return mask_input

def calc_mse(rec_img, image, crop=False):
    mse = nn.MSELoss()
    if crop:
        rec_img[:,:,31:47,21:42] = torch.zeros_like(rec_img[:,:,31:47,21:42])
        image[:,:,31:47,21:42] = torch.zeros_like(image[:,:,31:47,21:42])
    mse_loss = mse(rec_img, image)

    return mse_loss

def train(args, model, optimizer):
    dataset = iter(sample_data(args.path, args.batch, args.img_size))
    n_bins = 2. ** args.n_bits
    beta = 0
    #mse_loss = nn.MSELoss()

    save_path = f'./result/sample_{args.name}/'
    if (not os.path.exists(save_path)):
        os.makedirs(save_path)

    fo = open(save_path+'log_loss.csv', mode='w')
    writecsv = csv.writer(fo)
    writecsv.writerow(['n_sample','loss', 'logp', 'logdet', 'logc', 'rec'])

    z_sample = []
    z_shapes = calc_z_shapes(3, args.img_size, args.n_flow, args.n_block)
    for z in z_shapes:
        z_new = torch.randn(args.n_sample, *z) * args.temp
        z_sample.append(z_new.to(device))

    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            image, _ = next(dataset)
            image = image.to(device)

            if i == 0:
                with torch.no_grad():
                    x_sample = image
                    log_p, logdet, log_c, _, _ = model(image + torch.rand_like(image) / n_bins)

                    continue
            if i == 1:
                with torch.no_grad():
                    f_sample = image
                    continue

            else:
                log_p, logdet, log_c, z_outs, c_imgs = model(image + torch.rand_like(image) / n_bins)

            logdet = logdet.mean()

            loss, log_p, log_det, log_c = calc_loss(log_p, logdet, log_c, beta, args.img_size, n_bins)
            #model.zero_grad()
            #loss.backward()

            z_recs = []
            for j in range(4):
                if j == 3:
                    z_recs.append(z_outs[j])
                else:
                    z_recs.append(torch.randn_like(z_outs[j])*0.7)

            rec_img = model.reverse(z_recs, c_imgs, transform=True)
            #f_img = Mask(image)
            #loss_mse = mse_loss(rec_img, f_img).mean()
            loss_mse = calc_mse(rec_img, image)
            loss = loss + loss_mse
            model.zero_grad()
            loss.backward()

            _ = clip_grad_norm_(model.parameters(), 5)
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = args.lr
            optimizer.param_groups[0]['lr'] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f'Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; logC: {log_c.item():.5f}; rec: {loss_mse.item():.5f}'
            )
            writecsv.writerow([(i)*args.batch, loss.item(), log_p.item(), log_det.item(), log_c.item()])

            if i == 10:
                with torch.no_grad():
                    #f_sample = image
                    _, _, _, rz_outs, rc_imgs = model(x_sample)
                    utils.save_image(
                        model_single.reverse(rz_outs, rc_imgs, reconstruct=True).cpu().data,
                        save_path+'ref_lip.png',
                        normalize=True,
                        nrow=5,
                        range=(-0.5, 0.5),
                    )

            if (i+1) % 100 == 0:
                with torch.no_grad():
                    _, _, _, fz_outs, fc_imgs = model(f_sample)
                    _, _, _, rz_outs, rc_imgs = model(x_sample)
                    fz_recs = []
                    for j in range(4):
                        if j == 3:
                            fz_recs.append(fz_outs[j])
                        else:
                            fz_recs.append(torch.randn_like(fz_outs[j]))
                    
                    utils.save_image(
                        model_single.reverse(fz_recs, fc_imgs, transform=True).cpu().data,
                        save_path+f'{str(i + 1).zfill(6)}_r.png',
                        normalize=True,
                        nrow=5,
                        range=(-0.5, 0.5),
                    )
                    utils.save_image(
                        model_single.reverse(fz_outs, rc_imgs, transform=True).cpu().data,
                        save_path+f'{str(i + 1).zfill(6)}_t.png',
                        normalize=True,
                        nrow=5,
                        range=(-0.5, 0.5),
                    )
                    utils.save_image(
                        model_single.reverse(z_sample, rc_imgs).cpu().data,
                        save_path+f'{str(i + 1).zfill(6)}_s.png',
                        normalize=True,
                        nrow=5,
                        range=(-0.5, 0.5),
                    )

            if (i+1) % 10000 == 0:
                torch.save(
                    model.state_dict(), f'checkpoint/mi-flow_model_{args.name}_{str(i + 1).zfill(6)}.pt'
                )
                torch.save(
                    optimizer.state_dict(), f'checkpoint/mi-flow_optim_{args.name}_{str(i + 1).zfill(6)}.pt'
                )


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    model_single = Glow(
        3, args.n_flow, args.n_block, affine=args.affine, conv_lu=not args.no_lu
    )
    #model = nn.DataParallel(model_single)
    model = model_single
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(args, model, optimizer)
