import numpy as np
import cv2
import os


def trans(srcpath, dstpath):
    names = os.listdir(srcpath)
    for name in names:
        print(name)
        srcfile = os.path.join(srcpath, name)
        img = cv2.imread(srcfile)
        h, w = img.shape[0:2]
        writefile = os.path.join(dstpath, name)
        if h > w:
            M = cv2.getRotationMatrix2D((w//2, h//2), 90, 1)
            imgroataion = cv2.warpAffine(img, M, (h, w))
            cv2.imwrite(writefile, imgroataion)
        else:
            cv2.imwrite(writefile, img)

if __name__ == '__main__':
    srcpath = '/defaultShare/share/wujl/83/classibox/hfdata/traindata/hanfengv1/train/0'
    dstpath = '/defaultShare/share/wujl/83/classibox/hfdata/traindata/hanfengv2/train/0'
    os.makedirs(dstpath, exist_ok=True)
    trans(srcpath, dstpath)
