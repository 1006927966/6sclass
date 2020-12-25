import cv2
import numpy as np
import json
import os
import torchvision.models as models
import torch.nn as nn
import torchvision
from PIL import Image
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def make_model(key='resnet50'):
    model = models.__dict__[key](pretrained=False)
    fc_feature = model.fc.in_features
    model.fc = nn.Linear(fc_feature, 2)
    return model


def infer(img, loadpath):
    transforms = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
    img = Image.fromarray(img)
    img = transforms(img)
    img = img.unsqueeze(0)
    statedict = torch.load(loadpath, map_location='cpu')
    model = make_model()
    model.load_state_dict(statedict)
    print('[*]! loadding is true!!')
    model.cuda()
    model.eval()
    with torch.no_grad():
        img = img.cuda()
        pre = model(img)
    pre = torch.softmax(pre, 1)
    pre = pre.cpu().numpy()
    pre = pre[0]
    label = np.argmax(pre)
    return label


# poly2box
def poly2box(poly):
    xcoords = poly[:, 0]
    sorts = np.argsort(xcoords)
    lefttopx = poly[sorts[0]][0]
    rightbotomx = poly[sorts[-1]][0]
    ycoords = poly[:, 1]
    sorts = np.argsort(ycoords)
    lefttopy = poly[sorts[0]][1]
    rightbotomy = poly[sorts[-1]][1]
    return np.array([[int(lefttopx), int(lefttopy)], [int(rightbotomx), int(rightbotomy)]])

# 【x ,y】
def parse_json(jsonfile):
    with open(jsonfile) as f:
        line = f.readline()
    results = json.loads(line)
    polys = []
    print(results)
    for result in results:
        content = json.loads(result['content'])
        coords = content["shape"]
        coners = []
        for coord in coords:
            coners.append([int(coord['x']), int(coord['y'])])
        polys.append(np.array(coners))
    return polys


def croppic(imgname, jsondir, imgdir):
    if 'jpeg' in imgname:
        jsonname = imgname[:-5] + '.json'
    elif 'jpg' in imgname:
        jsonname = imgname[:-4] + '.json'
    else:
        return 0
    imgpath = os.path.join(imgdir, imgname)
    jsonpath = os.path.join(jsondir, jsonname)
    img = cv2.imread(imgpath)
    polys = parse_json(jsonpath)
    subimgs = []
    for poly in polys:
        box = poly2box(poly)
        subimg = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        subimgs.append(subimg)
    return subimgs


def drawresult(imgname, polys, labels):
    img = cv2.imread(imgname)
    n = len(polys)
    tag = 1
    for i in range(n):
        poly = polys[i]
        if labels[i] == 1:
            color = (0, 0, 255)
        elif labels[i] == 0:
            tag = 0
            color = (0, 255, 0)
        cv2.line(img, (poly[0][0], poly[0][1]), (poly[1][0], poly[1][1]), color, 5)
        cv2.line(img, (poly[1][0], poly[1][1]), (poly[2][0], poly[2][1]), color, 5)
        cv2.line(img, (poly[2][0], poly[2][1]), (poly[3][0], poly[3][1]), color, 5)
        cv2.line(img, (poly[3][0], poly[3][1]), (poly[0][0], poly[0][1]), color, 5)
    return img, tag


if __name__ == '__main__':
    savedir = '/defaultShare/share/wujl/83/classibox/hfdata/valzq/resultwh'
    savedraw = '/defaultShare/share/wujl/83/classibox/hfdata/valzq/drawwh'
    print('draw')
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedraw, exist_ok=True)
    loadpath = '/defaultShare/share/wujl/83/hfmodels/v2/model_20.pth'
    imgdir = '/defaultShare/share/wujl/83/classibox/hfdata/valzq/img'
    jsondir = '/defaultShare/share/wujl/83/classibox/hfdata/valzq/1054'
    imgnames = os.listdir(imgdir)
    for imgname in imgnames:
        print(imgname)
        if 'jpg' in imgname:
            jsonname = imgname[:-4] + '.json'
        elif 'jpeg' in imgname:
            jsonname = imgname[:-5] + '.json'
        jsonfile = os.path.join(jsondir, jsonname)
        polys = parse_json(jsonfile)
        imgpath = os.path.join(imgdir, imgname)
        subimgs = croppic(imgname, jsondir, imgdir)
        labels = []
        for subimg in subimgs:
            label = infer(subimg, loadpath)
            labels.append(label)
        cv2.imwrite(os.path.join(savedir, imgname), subimg)
        try:
            img, tag = drawresult(imgpath, polys, labels)
            drawpath = os.path.join(savedraw, '{}_'.format(tag) + imgname)
            cv2.imwrite(drawpath, img)
        except:
            continue





