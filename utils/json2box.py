import json
import os
import numpy as np
import cv2


# result key is taskImageId content:shape:[x, y]
def parse_json(jsonpath):
    with open(jsonpath, 'r', encoding='gbk') as f:
        jsn = f.read()
        jsdic = json.loads(jsn)
    results = jsdic['imageResults']
    imgs = jsdic['images']
    polysdic = {}
    for img in imgs:
        id = img['id']
        name = os.path.split(img['filePath'])[-1]
        polys = []
        labels = []
        for result in results:
            if result['taskImageId'] == id:
                label = result['label']
                content = json.loads(result['content'])
                coords = content["shape"]
                coners = []
                for coord in coords:
                    coners.append([int(coord['x']), int(coord['y'])])
                polys.append(np.array(coners))
                labels.append(label)
        polysdic[name] = [polys, labels]
    return polysdic


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


def polydic2boxdic(jsonpath):
    polydic = parse_json(jsonpath)
    boxdic = {}
    for key in polydic.keys():
        boxes = []
        polys, labels = polydic[key]
        for poly in polys:
            box = poly2box(poly)
            boxes.append(box)
        boxdic[key] = [boxes, labels]
    return boxdic


def cropbybox(boxdic, segpath, dstpath):
    picnames = os.listdir(segpath)
    for picname in picnames:
        if picname not in boxdic.keys():
            continue
        boxes = boxdic[picname]
        segimgpath = os.path.join(segpath, picname)
        img = cv2.imread(segimgpath)
        for box in boxes[0]:
            sliceimg = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            savepath = os.path.join(dstpath, '{}_{}_{}_{}_{}.jpg'.format(picname, box[0][0], box[0][1], box[1][0], box[1][1]))
            cv2.imwrite(savepath, sliceimg)
            print(savepath)


def boxdic2txt(savepath, boxdic):
    for key in boxdic.keys():
        txtname = key.split('.')[0] + '.txt'
        txtpath = os.path.join(savepath, txtname)
        with open(txtpath, 'a') as f:
            boxes, labels = boxdic[key]
            for i in range(len(labels)):
                label = labels[i]
                if label in ['hanfeng', 'han']:
                    index = 0
                elif label in ['wuhf', 'weihan']:
                    index = 1
                elif label in ['dianhan']:
                    index = 2
                else:
                    print('no this !!!!!')


def drawrectangle(boxdic, origindir, savedir):
    names = os.listdir(origindir)
    for name in names:
        picpath = os.path.join(origindir, name)
        img = cv2.imread(picpath)
        if name not in boxdic.keys():
            continue
        print(picpath)
        boxes = boxdic[name]
        for box in boxes:
            print(box[0])
            print(box[1])
            cv2.rectangle(img, (box[0][0], box[0][1]), (box[1][0], box[1][1]), (255, 0, 0), 3)
        savepath = os.path.join(savedir, name)
        cv2.imwrite(savepath, img)


if __name__ == '__main__':
    imgdir = '/defaultShare/share/wujl/83/seg/datav3/imgs'
    jsons = '/defaultShare/share/wujl/83/seg/datav3/jsons'
    selectjsons = os.listdir(jsons)[0:3]
    savedir = '/defaultShare/share/wujl/83/classibox/hanfeng/1'
    for jsonname in selectjsons:
        jsonpath = os.path.join(jsons, jsonname)
        print(jsonpath)
        boxdic = polydic2boxdic(jsonpath)
        cropbybox(boxdic, imgdir, savedir)