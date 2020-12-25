import json
import os
import numpy as np
import cv2


def distance(point1, point2):
    dist = (point1[0]-point2[0])**2 + (point1[1] - point2[1])**2
    return dist


#point [x, y]
def giveorder(points):
    n = points.shape[0]
    ddic = {}
    for i in range(n):
        for j in range(i+1, n):
            ddic[[i,j]] = distance(points[i], points[j])
    pairs_list = []
    dist_list = []
    for key in ddic.keys():
        pairs_list.append(key)
        dist_list.append(ddic[key])
    indexs = np.argsort(np.array(dist_list))
    usepair = pairs_list[indexs[-3]]
    




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








