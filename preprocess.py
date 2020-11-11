from PIL import Image
import os
import numpy as np
import dlib
import matplotlib.pylab as plt

# argument parser setting
import argparse
parser = argparse.ArgumentParser(description='Reference image preprocessing for face translation')
parser.add_argument('--verbose', default=True, type=bool, help='showing the status of the transform procedure')
parser.add_argument('path', metavar='PATH', type=str, help='Path of reference image directory')

def lip_rect(img, detector, predictor, space=10):
    dets = detector(img, 1)
    for i, det in enumerate(dets):
        shape = predictor(img, det)
        if i > 0:
            shape = None
    try:
        left = shape.part(48).x
        right = shape.part(54).x
        up = shape.part(50).y
        down = shape.part(57).y
    except:
        left = l
        right = r
        up = u
        down = d
    return left, right, up, down


def main(path, detector, predictor):

    name = path.split("/")[-1].split(".")[0]
    img = Image.open(path)
    #img = img.resize((486,380), Image.BILINEAR)
    img_n = np.array(img).astype('uint8')[:,:,:3]
    l, r, u, d = lip_rect(img_n, detector, predictor) 
    print("original lip location:",l, r, u, d)
    # plt.imshow(img_n[u:d,l:r])

    width = 43
    bili = 43/(r-l)
    if bili > 1:
        print("image size is too small")
    else:
        w = int(img_n.shape[0]*bili)
        h = int(img_n.shape[1]*bili)
        img = img.resize((h, w), Image.BILINEAR)
        img =  np.array(img).astype('uint8')[:,:,:3]
        l, r, u, d = lip_rect(img, detector, predictor)
        print("reshape lip location:",l, r, u, d)

        y = int((u+d)/2)
        x = int((l+r)/2)
        #print(img.shape, y, x)
        if (y < 160 or x < 128 or (x+128 > img.shape[1]) or (y+96 > img.shape[0])):
            y1 = max(0, y-160)
            y2 = min(img.shape[0], y+96)
            x1 = max(0, x-128)
            x2 = min(img.shape[1], x+128)
            #print(y1, y2, x1, x2)
            if (y2-y1 > x2-x1):
                shift = int(((y2-y1)-(x2-x1))/2)
                #print(shift)
                img_cent = img[y1+shift:y2-shift, x1:x2]
            elif (y2-y1 < x2-x1):
                shift = int(((x2-x1)-(y2-y1))/2)
                #print(shift)
                img_cent = img[y1:y2, x1+shift:x2-shift]
            else:
                img_cent = img[y1:y2, x1:x2]
            im = Image.fromarray(img_cent)
            im = im.resize((256,256), Image.BILINEAR)
            print("crop image shape: ",img_cent.shape, "-> (256,256, 3)")
            im.save(f"./image/ref_img/{name}_256.jpg")

        else:
            img_cent = img[y-160:y+96, x-128:x+128]
            print("crop image shape: ", img_cent.shape)
            im = Image.fromarray(img_cent)
            im.save(f"./image/ref_img/{name}_256.jpg")
        
        print(f"save figure in ./image/ref_img/{name}_256.jpg")


if __name__ == "__main__":
    args = parser.parse_args()

    predictor_path = "./ad-flow/checkpoint/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    main(args.path, detector, predictor)