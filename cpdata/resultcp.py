import shutil
import os

map_list = [1, 3, 5, 6, 8, 12, 13, 14, 16, 18]

def cp(txtpath, origindir, savedir, maplist):
    os.makedirs(savedir, exist_ok=True)
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        factors = line.split(',')
        label = int(factors[-1])
        if label in maplist:
            name = factors[0]
            readfile = os.path.join(origindir, name)
            writefile = os.path.join(savedir, name)
            shutil.copy(readfile, writefile)


if __name__ == '__main__':
    path = '/VisualGroup/share/wujl/83/test/bg_test/txtpath/croppre_0.4_0.1_50.txt'
    origindir = '/VisualGroup/share/wujl/83/test/bg_test/croppic/croppre_0.4_0.1'
    savedir = '/VisualGroup/share/wujl/83/test/bg_test/croppic/badcase'
    cp(path, origindir, savedir, map_list)