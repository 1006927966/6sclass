import cv2
import os

rootdir = '/defaultShare/share/wujl/83/ddb_22/train'
classes = os.listdir(rootdir)

for classs in classes:
    # if int(classs) != 8:
    #      continue
    subdir = os.path.join(rootdir, classs)
    names = os.listdir(subdir)
    for name in names:
        path = os.path.join(subdir, name)
        print(path)
        a = cv2.imread(path)

            # print('bd')
            # os.remove(path)
            # continue
