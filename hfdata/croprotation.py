import cv2
import numpy as np
import math
from PIL import Image
import os
import json


# poly should be np.array
def getpoints(poly):
    rect = cv2.minAreaRect(poly)
    center, size, angle = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    cbox = cv2.boxPoints(rect)
    height = math.hypot(cbox[0][0] - cbox[1][0], cbox[0][1] - cbox[1][1])
    width = math.hypot(cbox[0][0] - cbox[3][0], cbox[0][1] - cbox[3][1])
    if height > width:
        angle = angle + 90
        size = (size[1], size[0])
    size = (size[0], int(size[1]*1.1))
    return center, size, angle


def drawbox(img, bbox):
    color = (0, 0, 255)
    for box in bbox:
        cv2.line(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), color, 10)
        cv2.line(img, (box[1][0], box[1][1]), (box[2][0], box[2][1]), color, 10)
        cv2.line(img, (box[2][0], box[2][1]), (box[3][0], box[3][1]), color, 10)
        cv2.line(img, (box[3][0], box[3][1]), (box[0][0], box[0][1]), color, 10)
    return img


def crop(img, poly):
    center, size, angle = getpoints(poly)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1)
    img_rot = cv2.warpAffine(img, M, (w, h))
    img_crop = cv2.getRectSubPix(img_rot, size, center)
    return img_crop


def parse_json(jsondir, jpeg=True):
    polydic = {}
    jsonpaths = os.listdir(jsondir)
    for jsonpath in jsonpaths:
        jsonfile = os.path.join(jsondir, jsonpath)
        print(jsonfile)
        with open(jsonfile) as f:
            line = f.readline()
        results = json.loads(line)
        polys = []
        for result in results:
            content = json.loads(result['content'])
            coords = content["shape"]
            coners = []
            for coord in coords:
                coners.append([int(coord['x']), int(coord['y'])])
            polys.append(np.array(coners))
        if jpeg:
            polydic[jsonpath[:-4] + 'jpeg'] = polys
        else:
            polydic[jsonpath[:-4] + 'jpg'] = polys
    return polydic


def cropbyjson(jsondir, imgdir, savedir):
    polydic = parse_json(jsondir, jpeg=False)
    names = os.listdir(imgdir)
    for name in names:
        if name in polydic.keys():
            print(name)
            imgpath = os.path.join(imgdir, name)
            img = cv2.imread(imgpath)
            polys = polydic[name]
            polynum = len(polys)
            for i in range(polynum):
                poly = polys[i]
                img_crop = crop(img, poly)
                savepath = os.path.join(savedir, '{}_'.format(i)+name)
                cv2.imwrite(savepath, img_crop)

if __name__ == '__main__':
    jsondir = '/defaultShare/share/wujl/83/classibox/hfdata/valzq/1054'
    imgdir = '/defaultShare/share/wujl/83/classibox/hfdata/valzq/img'
    savedir = '/defaultShare/share/wujl/83/classibox/hfdata/croprote/hfzqv2'
    os.makedirs(savedir, exist_ok=True)
    cropbyjson(jsondir, imgdir, savedir)