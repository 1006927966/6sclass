import cv2
import numpy as np
import os


def corpbyanchor(img, size):
    imglist = []
    h, w = img.shape[0:2]
    anchorx = w//size
    anchory = h//size
    if anchorx < 100 or anchory < 100:
        return imglist
    for i in range(size*size):
        xcap = i%size
        ycap = i//size
        beginx = xcap * anchorx
        endx = (xcap+1)*anchorx
        beginy = ycap * anchory
        endy = (ycap + 1)* anchory
        subimg = img[beginy:endy, beginx:endx]
        imglist.append(subimg)
    return imglist


def cropx(dirpath, size, savedir):
    names = os.listdir(dirpath)
    for name in names:
        print(name)
        filepath = os.path.join(dirpath, name)
        img = cv2.imread(filepath)
        imglist = corpbyanchor(img, size)
        if len(imglist) == 0:
            continue
        for i in range(len(imglist)):
            print(i)
            savefile = os.path.join(savedir, name[:-4]+'{}.jpg'.format(i))
            cv2.imwrite(savefile, imglist[i])

if __name__ == '__main__':
    dirpath = '/defaultShare/share/wujl/83/online_data/ddb_train_bg_v2'
    savepath = '/defaultShare/share/wujl/83/online_data/ddb_crop'
    os.makedirs(savepath, exist_ok=True)
    cropx(dirpath, 4, savepath)
