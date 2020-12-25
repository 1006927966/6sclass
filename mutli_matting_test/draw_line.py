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
        tags.append(obj.find("name").text)
        xmlbox = obj.find('bndbox')
        box.append(int(float(xmlbox.find("xmin").text)))
        box.append(int(float(xmlbox.find("ymin").text)))
        box.append(int(float(xmlbox.find("xmax").text)))
        box.append(int(float(xmlbox.find("ymax").text)))
        boxes.append(box)
    return boxes, tags

def drawline(picpath, boxes, savedir):
    name = picpath.split('/')[-1]
    img = cv2.imread(picpath)
    for box in boxes:
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
    savepath = os.path.join(savedir, name)
    cv2.imwrite(savepath, img)

if __name__ == '__main__':
    rootdir = '/defaultShare/share/wujl/83/online_data/recalldir'
    savedir = os.path.join(rootdir, 'line')
    os.makedirs(savedir, exist_ok=True)
    xmlspath = os.path.join(rootdir, 'Annotations')
    picspath = os.path.join(rootdir, 'JPEGImages')
    picnames = os.listdir(picspath)
    for picname in picnames:
        ip = picname[:-5]
        picpath = os.path.join(picspath, picname)
        xmlname = ip + '.xml'
        xmlpath = os.path.join(xmlspath, xmlname)
        boxes, tags = parse_xml(xmlpath)
        if len(boxes) == 0:
            continue
        drawline(picpath, boxes, savedir)