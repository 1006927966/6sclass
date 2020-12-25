import json
import os
import numpy as np
import cv2


# result key is taskImageId content:shape:[x, y]
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


def polydic2boxdic(jsondir):
    polydic = parse_json(jsondir, jpeg=False)
    boxdic = {}
    for key in polydic.keys():
        boxes = []
        polys = polydic[key]
        for poly in polys:
            box = poly2box(poly)
            boxes.append(box)
        boxdic[key] = boxes
    return boxdic


def cropbybox(boxdic, segpath, dstpath):
    picnames = os.listdir(segpath)
    for picname in picnames:
        if picname not in boxdic.keys():
            continue
        boxes = boxdic[picname]
        segimgpath = os.path.join(segpath, picname)
        img = cv2.imread(segimgpath)
        for box in boxes:
            sliceimg = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
            savepath = os.path.join(dstpath, '{}_{}_{}_{}_{}.jpg'.format(picname, box[0][0], box[0][1], box[1][0], box[1][1]))
            cv2.imwrite(savepath, sliceimg)
            print(savepath)


if __name__ == '__main__':
    jsondir = '/defaultShare/share/wujl/83/classibox/hfdata/origindata/hfv3/hfv3'
    imgpath = '/defaultShare/share/wujl/83/classibox/hfdata/origindata/hfv3/imgv3/'
    dstpath = '/defaultShare/share/wujl/83/classibox/hfdata/traindata/1'
    os.makedirs(dstpath, exist_ok=True)
    jsonpathnames = os.listdir(jsondir)
    for jsonpathname in jsonpathnames:
        jsonpath = os.path.join(jsondir, jsonpathname)
        boxdic = polydic2boxdic(jsonpath)
        print(boxdic)
        cropbybox(boxdic, imgpath, dstpath)



