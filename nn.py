import os
import shutil

def cp(origindir, savedir):
    classimap = [[0], [1,2], [3], [4, 5], [6, 7], [8], [9, 24], [10], [11], [12],
                 [12, 14, 15], [16, 17], [18, 19], [20], [21], [22], [23]]
    labels = os.listdir(origindir)
    for label in labels:
        originpath = os.path.join(origindir, label)
        names = os.listdir(originpath)
        for name in names:
            readfile = os.path.join(originpath, name)
            for i in range(len(classimap)):
                if int(label) in classimap[i]:
                    savepath = os.path.join(savedir, '{}'.format(i))
                    os.makedirs(savepath, exist_ok=True)
                    writefile = os.path.join(savepath, name)
                    print(writefile)
                    shutil.copy(readfile, writefile)



if __name__ == '__main__':
    origindir = '/VisualGroup/share/wujl/83/ddb_25/train'
    savedir = '/VisualGroup/share/wujl/83/ddb_label/train'
    cp(origindir, savedir)