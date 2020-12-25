import cv2
import os

path = '/VisualGroup/share/wujl/83/ddb_25/train'
names = os.listdir(path)
print(names)
for name in names:
    subpath = os.path.join(path, name)
    subnames = os.listdir(subpath)
    for subname in subnames:
        picfile = os.path.join(subpath, subname)
        print(name)
        print(picfile)
        img = cv2.imread(picfile)
