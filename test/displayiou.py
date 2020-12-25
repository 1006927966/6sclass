import os
import cv2
from xml.etree import ElementTree as ET

def parse_xml(xmlpath):
    boxes = []
    trees = ET.parse(xmlpath)
    root = trees.getroot()
    for obj in root.iter("object"):
        box = []
        xmlbox = obj.find("bndbox")
        box.append(int(float(xmlbox.find("xmin").text)))
        box.append(int(float(xmlbox.find("ymin").text)))
        box.append(int(float(xmlbox.find("xmax").text)))
        box.append(int(float(xmlbox.find("ymax").text)))
        boxes.append(box)
    return boxes

def parsepre(id, cropdir):
    cropnames = os.listdir(cropdir)
    boxes = []
    for cropname in cropnames:
        box = []
        ip = cropname.split('_')[0] + '_' + cropname.split('_')[1]
        if ip == id:
            box.append(int(float(cropname.split('_')[-4])))
            box.append(int(float(cropname.split('_')[-3])))
            box.append(int(float(cropname.split('_')[-2])))
            box.append(int(float(cropname.split('_')[-1][:-4])))
        if len(box) != 0:
            boxes.append(box)
    return boxes



def draw(imgname, rootdir, cropdir, xmldir, savedir):
    os.makedirs(savedir, exist_ok=True)
    if 'jpg' in imgname:
        id = imgname[:-4]
    else:
        id = imgname[:-5]
    imgpath = os.path.join(rootdir, imgname)
    img = cv2.imread(imgpath)
    xmlpath = os.path.join(xmldir , id+'.xml')
    if not os.path.exists(xmlpath):
        return 0
    boxes = parse_xml(xmlpath)
    print(xmlpath)
    preboxes = parsepre(id, cropdir)
    if len(preboxes) == 0:
        return 0
    print(imgname)
    print(preboxes)
    for box in boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 5)
    for prebox in preboxes:
        print(prebox)
        cv2.rectangle(img, (prebox[0], prebox[1]), (prebox[2], prebox[3]), (255,0,0), 5)
    savepath = os.path.join(savedir, imgname)
    cv2.imwrite(savepath, img)
    return 1


if __name__ == '__main__':
    rootdir = '/VisualGroup/share/wujl/83/test/9_21/originpic/JPEGImages'
    cropdir = '/VisualGroup/share/wujl/83/test/9_21/maskcrop/croprecall_0.1_0.1_0.1'
    xmldir = '/VisualGroup/share/wujl/83/test/9_21/originpic/Annotations'
    savedir = '/VisualGroup/share/wujl/83/test/9_21/testcrop/0.1iou0.1'
    names = os.listdir(rootdir)
    for name in names:
        a = draw(name, rootdir, cropdir, xmldir, savedir)


