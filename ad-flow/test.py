#!/usr/bin/env python
# -*- coding: utf-8 -*-
#__author__ = "Sheng-Je Huang"
#__version__ = "1.0.0"

from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import imageio
from PIL import Image
from skimage import img_as_ubyte

import torch
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

from dataset import FaceDataset
from model import Glow as glow
from model import feature_encoder, seq_classifier        # standard model setting
#from AD_model import feature_encoder, seq_classifier    # ablation study model setting

from scipy.sparse.linalg import spsolve
import scipy.sparse

# argument parser setting
import argparse
parser = argparse.ArgumentParser(description='Attribute decomposition flow testing - image transform')
parser.add_argument('--gdir', default='./checkpoint/glow_checkpoint.pt', type=str, help='glow model weight directory')
parser.add_argument('--edir', default='./checkpoint/ad-flow_total_checkpoint.tar',type=str, help='feature encoder weight directory')
parser.add_argument('--poisson', default=True, type=bool, help='using poisson blur for image fusion')
parser.add_argument('--source', default='trump', type=str, help='source image name (trump/ellen)')
parser.add_argument('--verbose', default=True, type=bool, help='showing the status of the transform procedure')
parser.add_argument('refname', metavar='NAME', type=str, help='Name of reference image with size 256*256')

transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ]
)

def laplacian_matrix(n, m):   
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A

# Poisson Blur
def PoissonBlur(size=256):
    mask = torch.zeros([size,size])
    ratio = size/256
    a = int(140*ratio)
    b = int(188*ratio)
    # c = int(88*ratio)
    # d = int(168*ratio)
    c = int(90*ratio)
    d = int(166*ratio)
    mask[a:b,c:d] = torch.ones_like(mask[a:b,c:d])

    mat_A = laplacian_matrix(size, size)
    laplacian = mat_A.tocsc()

    x_range = size
    for y in range(1, x_range-1):
        for x in range(1, x_range-1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0
    mat_A = mat_A.tocsc()
    return mat_A, mask, laplacian

def main(args, model2, f_encoder):
    # transform
    print("[info] Start transform ...")
    y_min = 0
    x_min = 0
    name = args.refname
    t_dir = f'../image/ref_img/{name}_256.jpg'
    target = np.array(Image.open(t_dir).convert('RGB'))
    face = transform(Image.open(t_dir).convert('RGB')).unsqueeze(0)
    y_max = target.shape[1]
    x_max = target.shape[0]
    y_range = y_max - y_min
    x_range = x_max - x_min

    mat_A, mask, laplacian = PoissonBlur()
    mask_flat = mask.flatten()
    poisson = args.poisson
    
    s_name = args.source
    path = f'../image/source_img/{s_name}/'
    Dir = os.listdir(path)
    Dir.sort()
    mix_imgs = []

    save_path = f'./result/target_img/{s_name}/{name}/'
    if (not os.path.exists(save_path)):
        os.makedirs(save_path)

    for n, filename in enumerate(Dir):
        if filename[0] == ".":
            continue
        file_dir = os.path.join(path,filename)
        m_img = Image.open(file_dir).convert('RGB')

        if poisson:
            if m_img.height != y_max or m_img.width != x_max:
                m_img = m_img.resize((x_max,y_max), Image.BILINEAR)
            source = np.array(m_img)

            for channel in range(source.shape[2]):
                source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
                target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()        

                # inside the mask:
                # \Delta f = div v = \Delta g       
                alpha = 1
                mat_b = laplacian.dot(source_flat)*alpha

                # outside the mask:
                # f = t
                mat_b[mask_flat == 0] = target_flat[mask_flat == 0]

                x = spsolve(mat_A, mat_b)    
                x = x.reshape((y_range, x_range))
                x[x > 255] = 255
                x[x < 0] = 0
                x = x.astype('uint8')

                target[y_min:y_max, x_min:x_max, channel] = x
            mix_img = transform(Image.fromarray(target)).unsqueeze(0)
        else:
            mix_img = transform(m_img).unsqueeze(0)
        
        # AD-flow
        with torch.no_grad():
            _, _, zf_img = model2(face)
            _, _, zl_imgs = model2(mix_img)
            zf_face, zf_lip, zf_mean = f_encoder(zf_img)
            zl_faces, zl_lips, zl_mean = f_encoder(zl_imgs)
        z_rec = []
        for i in range(4):
            zf = zf_face[i]
            zl = zl_lips[i]
            #z_ori.append(zf + zr_lips[i][9:10])
            z_rec.append(zf + zl)

        with torch.no_grad():
            utils.save_image(
                model2.reverse(z_rec, reconstruct=True).cpu().data,
                save_path+f'{n:03d}.png', normalize=True, nrow=1, range=(-0.5, 0.5)
                )
            #./data/gif/{s_name}/AD-cent/AD-{name}/{n:03d}.png
        if (args.verbose):        
            print(f"save {n:03d}.png", end=" ", flush=True)

       # if n == 5:
       #     break
    print(f"\n[info] Finish! Please check result in result/target_img/{s_name}/{name}")


if __name__ == '__main__':
    args = parser.parse_args()

    # load model
    print("[info] Load glow model ...")
    model = glow(3, 32, 4, False, True)
    model = model.to("cpu")
    model.load_state_dict(torch.load(args.gdir))
    #print("Glow model parameters:", sum(p.numel() for p in model2.parameters()))

    print("[info] Load encoder model ...")
    checkpoint = torch.load(args.edir)
    f_encoder = feature_encoder()
    #s_classifier = seq_classifier(2720)
    f_encoder.load_state_dict(checkpoint['f_encoder'])
    #s_classifier.load_state_dict(checkpoint['s_class'])
    #print("f encoder parameters:", sum(p.numel() for p in f_encoder.parameters()))

    main(args, model, f_encoder)
