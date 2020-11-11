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

import random
from dataset import FaceDataset
from model import Glow, feature_encoder, seq_classifier              # standard model setting
# from AD_model import Glow, feature_encoder, seq_classifier         # ablation study model setting

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

parser = argparse.ArgumentParser(description='Talking face trainer')
parser.add_argument('--batch', default=1, type=int, help='batch size')
parser.add_argument('--iter', default=200000, type=int, help='maximum iterations')
parser.add_argument('--n_bits', default=5, type=int, help='number of bits')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--img_size', default=64, type=int, help='image size')
parser.add_argument('--temp', default=0.7, type=float, help='temperature of sampling')
parser.add_argument('--n_sample', default=20, type=int, help='number of samples')
parser.add_argument('--classifier', default=True, type=bool, help='sequential classification')
parser.add_argument('-n', '--filename', type=str, help='checkpoint recognized name')
# parser.add_argument('-l', '--loadfile', type=str, default=None, help='load training status')
parser.add_argument('-s', '--start_iter', default=0, type=int, help='start of iteration')
parser.add_argument('path', metavar='PATH', type=str, help='Path to image directory')

def sample_data(dataset, batch_size, image_size):

    #dataset = FaceDataset(path, transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            print("------ epoch end ------")
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=1
            )
            loader = iter(loader)
            yield next(loader)

def calc_w_shapes(img_size):
    weight = []
    for i in range(4):
        size = int(img_size/(2**(i+1)))
        w = torch.randn([size, size])
        w = w.to(device)
        weight.append(w)

    return weight

def train(args, Gmodel):
    transform = transforms.Compose(
        [
            transforms.Resize(args.img_size),
            transforms.CenterCrop(args.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
        ]
    )
    Fdataset = FaceDataset(args.path, transform=transform)
    dataset = iter(sample_data(Fdataset, args.batch, args.img_size))

    # weight = []
    # weight = calc_w_shapes(args.img_size)
    # W_dict = {"weight":weight}
    # optimizer = optim.Adam(weight, lr=args.lr)
    f_encoder = feature_encoder(mode='F')
    f_encoder = f_encoder.to(device)
    s_classifier = seq_classifier(2720) # (16 + 64 + 256 + 1024) * 2
    s_classifier = s_classifier.to(device)

    f_optimizer = optim.Adam(f_encoder.parameters(), lr=args.lr)
    s_optimizer = optim.Adam(s_classifier.parameters(), lr=0.0001)

    loadFilename = args.start_iter
    if loadFilename:
        print("load weight and optim from iter:", loadFilename, end=" ")
        checkpoint = torch.load(f"./checkpoint/ad-flow_{args.filename}_{str(args.start_iter).zfill(6)}.tar")
        f_encoder.load_state_dict(checkpoint['f_encoder'])
        f_optimizer.load_state_dict(checkpoint['f_optim'])
        print("OK")
    
    save_path = f'./image/sample_{args.filename}/'
    if (not os.path.exists(save_path)):
        os.makedirs(save_path)

    Loss = nn.MSELoss()
    Cross = nn.CrossEntropyLoss()
    pred_c1 = 0
    pred_c5 = 0
    
    print("[info] start training ...")
    with tqdm(range(args.iter)) as pbar:
        for i in pbar:
            loss_rec = 0
            loss_cen = 0
            loss_rec2 = 0
            #loss_rec3 = 0

            images, word, ID, seq = next(dataset)
            images = images[0].to(device)
            with torch.no_grad():
                _, _, z_imgs = Gmodel(images)

            shift = int(random.randint(2, len(seq)))

            z_faces, z_lips, z_mean = f_encoder(z_imgs)

            if i == 0:
                with torch.no_grad():
                    z_refer = z_imgs.copy()
                    p_faces, p_lips, p_mean = f_encoder(z_imgs)
                continue
            
            p_recs = []
            for j in range(4):
                zf = z_faces[j]
                zl = z_lips[j]
                zm = z_mean[j]

                # z_rec3 = zf + zl
                # loss_rec3 += Loss(z_rec3, z_imgs[j])

                # Sequential random pair loss
                zf_shift = torch.cat((zf[shift:], zf[:shift]), 0)
                z_rec = zf_shift + zl
                loss_rec += Loss(z_rec, z_imgs[j])

                # Structural-peceptual loss
                loss_cen += Loss(zf, z_mean[j].expand(zf.shape))

                # Cycle consistency loss
                p_rec2 = p_faces[j][0].expand(zf.shape) + zl
                pf = f_encoder.Fencoders[j](p_rec2)
                loss_rec2 += Loss(pf, p_mean[j].expand(zf.shape))

            loss = loss_cen + loss_rec + loss_rec2 #+ loss_class

            # Classification
            if args.classifier:
                s_optimizer.zero_grad()
                pred = s_classifier(z_lips, [len(seq)])
                label = torch.tensor([Fdataset.word_idx[word[0]]]).to(device)
                loss_class = Cross(pred, label)
                loss += loss_class

                # classifier result
                y = torch.topk(pred, k=5, dim=1)[1]
                t = Fdataset.word_idx[word[0]]
                if t in y:
                    pred_c5 += 1
                    if t == y[0][0]:
                        pred_c1 += 1

            f_optimizer.zero_grad()
            loss.backward()
            f_optimizer.step()

            if args.classifier:
                s_optimizer.step()
                pbar.set_description(
                    f'Loss: {loss.item():.5f}; rec: {loss_rec.item():.5f}; cen: {loss_cen.item():.5f}; rec2: {loss_rec2.item():.5f}; class: {loss_class.item():.5f}; pred1:{pred_c1/(i+1):.4f}, pred5:{pred_c5/(i+1):.4f}')
            else:
                pbar.set_description(
                    f'Loss: {loss.item():.5f}; rec: {loss_rec.item():.5f}; cen: {loss_cen.item():.5f}; rec2: {loss_rec2.item():.5f}'
                    ) #; class: {loss_class.item():.5f}; pred1:{pred_c1/(i+1):.4f}, pred5:{pred_c5/(i+1):.4f}
            
            if (i+1) % 200 == 0:
                with torch.no_grad():
                    seq_size = len(seq)
                    z_sample = []
                    zr_faces, zr_lips, zr_mean = f_encoder(z_refer)

                    for j in range(4):
                        zt = z_refer[j][:1]
                        zf = zr_faces[j][0].repeat(seq_size, 1, 1, 1)
                        zl = z_lips[j]
                        zr = zf + zl
                        zs = torch.cat((zt, zr, z_imgs[j][:1], z_imgs[j]), 0)
                        z_sample.append(zs)

                    utils.save_image(
                        Gmodel.reverse(z_sample, reconstruct=True).cpu().data,
                        save_path+f'{str(i + 1 + args.start_iter).zfill(6)}.png',
                        normalize=True,
                        nrow=seq_size+1,
                        range=(-0.5, 0.5),
                    )

            if (i+1) % 10000 == 0:
                save_dict = {
                    'iteration':i+1,
                    'f_encoder': f_encoder.state_dict(),
                    'f_optim' : f_optimizer.state_dict()
                }
                if args.classifier:
                    save_dict['s_class'] = s_classifier.state_dict()
                    save_dict['s_optim'] = s_optimizer.state_dict()
                torch.save(save_dict, f'./checkpoint/ad-flow_{args.filename}_{str(i + 1 + args.start_iter).zfill(6)}.tar')
            
            with torch.no_grad():
                p_faces, p_lips, p_mean = f_encoder(z_imgs)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    model = Glow(3, 32, 4, False, True)
    model = model.to(device)
    model.load_state_dict(torch.load("./checkpoint/glow_checkpoint.pt"))
    for param in model.parameters():
        param.requires_grad_(False)
    
    train(args, model)
