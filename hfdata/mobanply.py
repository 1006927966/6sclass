import cv2
import numpy as np
import json
import os


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


def drawmoban(mobanpath, polys, savedir):
    names = os.listdir(mobanpath)
    for name in names:
        print(name)
        mobanfile = os.path.join(mobanpath, name)
        img = cv2.imread(mobanfile)
        for poly in polys:
            print(poly[0])
            print(poly[1])
            cv2.line(img, (poly[0][0], poly[0][1]), (poly[1][0], poly[1][1]), (0,0,255), 5)
            cv2.line(img, (poly[1][0], poly[1][1]), (poly[2][0], poly[2][1]), (0, 0, 255), 5)
            cv2.line(img, (poly[2][0], poly[2][1]), (poly[3][0], poly[3][1]), (0, 0, 255), 5)
            cv2.line(img, (poly[3][0], poly[3][1]), (poly[0][0], poly[0][1]), (0, 0, 255), 5)
        writefile = os.path.join(savedir, name)
        cv2.imwrite(writefile, img)

if __name__ == '__main__':
    jsonfile = '/defaultShare/share/wujl/83/classibox/hfdata/mobantest/a.json'
    mobanpath = '/defaultShare/share/wujl/83/classibox/hfdata/mobantest/moban'
    savedir = '/defaultShare/share/wujl/83/classibox/hfdata/mobantest/drawmoban'
    os.makedirs(savedir, exist_ok=True)
    polys = parse_json(jsonfile)
    drawmoban(mobanpath, polys, savedir)
