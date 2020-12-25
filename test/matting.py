import numpy as np
import cv2
import os
import xml.etree.ElementTree as ET
#from lxml import etree, objectify

def parse_xml(xmlpath):
    boxes = []
    tags = []
    tree = ET.parse(xmlpath)
    root = tree.getroot()
    for obj in root.iter("object"):
        box = []
        name = obj.find("name").text
        print(name)
        if name == '物料占道':
            continue
        tags.append(name)
        xmlbox = obj.find('bndbox')
        box.append(float(xmlbox.find("xmin").text))
        box.append(float(xmlbox.find("ymin").text))
        box.append(float(xmlbox.find("xmax").text))
        box.append(float(xmlbox.find("ymax").text))
        boxes.append(box)
    return boxes, tags

def matting(boxes, img, rate):
    matting_imgs = []
    h, w = img.shape[:2]
    for box in boxes:
        y_pd = int((box[-1] - box[1])*rate) + 1
        x_pd = int((box[-2] - box[0])*rate) + 1
        y_min = int(box[1]) - y_pd
        y_max = int(box[-1]) + y_pd
        x_min = int(box[0]) - x_pd
        x_max = int(box[-2]) + x_pd

        if y_min <0 :
            y_min = 0
        if y_max > h:
            y_max = h
        if x_min < 0:
            x_min = 0
        if x_max>w:
            x_max = w
        matting_img = img[y_min:y_max, x_min:x_max]
        matting_imgs.append(matting_img)
    return matting_imgs


def matting_imgs(imgspath, xmlspath, savepath,  numconstraint=65000, name_bool=False, rate=0.1):
    imgnames = os.listdir(imgspath)
    count = 0
    for imgname in imgnames:
        if name_bool:
            if imgname not in name_bool:
                continue
        if count >= numconstraint:
            return 0
        if ('jpeg' not in imgname) and ('jpg' not in imgname):
            continue
        if 'jpeg' in imgname:
            id = imgname[:-5]
        if 'jpg' in imgname:
            id = imgname[:-4]

        imgpath = os.path.join(imgspath, imgname)
        xmlpath = os.path.join(xmlspath, id+'.xml')
        print(xmlpath)
        print(imgpath)
        if not os.path.exists(xmlpath):
            print('xml is not exist!!!!')
            continue
        img = cv2.imread(imgpath)
        boxes, tags = parse_xml(xmlpath)
        boxnum = len(boxes)
        if boxnum == 0:
            continue
        for i in range(boxnum):
            count += 1
            box = boxes[i]
            savefilename = '{}_{}_{}_{}_{}.jpg'.format(id, box[0], box[1], box[2], box[3])
            savefilefolder = os.path.join(savepath, 'badcase')
            os.makedirs(savefilefolder, exist_ok=True)
            savefilepath = os.path.join(savefilefolder, savefilename)
            matting_imgs = matting([box], img, rate)
            cv2.imwrite(savefilepath, matting_imgs[0])
        print('[*]! begin next pic')
    return 1





if __name__ == "__main__":
    xmlpath = '/VisualGroup/share/wujl/83/test/9_21/originpic/ct_det_xml/10.11.149.61_20200926100225.xml'
    imgpath = '/VisualGroup/share/wujl/83/test/9_21/originpic/JPEGImages/10.11.149.61_20200926100225.jpeg'
    savedir = '/VisualGroup/share/wujl/83/test/9_21/croppic/mtest'
    os.makedirs(savedir, exist_ok=True)
    img = cv2.imread(imgpath)
    boxes, tags = parse_xml(xmlpath)
    print(boxes)
    rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    for rate in rates:
        img_list =  matting(boxes, img, rate)
        subimg = img_list[0]
        savepath = os.path.join(savedir, '{}.jpg'.format(rate))
        cv2.imwrite(savepath, subimg)
    # imgspath = '/defaultShare/share/wujl/83/test/9_21/originpic/JPEGImages'
    # xmlspath = '/defaultShare/share/wujl/83/test/9_21/originpic/Annotations'
    # for rate in rates:
    #     savepath = '/defaultShare/share/wujl/83/test/9_21/croppic/crop_{}'.format(rate)
    #     os.makedirs(savepath, exist_ok=True)
    #     p = matting_imgs(imgspath, xmlspath, savepath, numconstraint=20000000, rate=rate)
    # print('end')


