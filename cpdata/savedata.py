import os
import shutil

def cp(path, savepath):
    firsts = os.listdir(path)
    for first in firsts:
        secondpath = os.path.join(path, first)
        seconds = os.listdir(secondpath)
        for second in seconds:
            thirdpath = os.path.join(secondpath, second)
            thirds = os.listdir(thirdpath)
            for third in thirds:
                fourthpath = os.path.join(thirdpath, third)
                fourths = os.listdir(fourthpath)
                for fourth in fourths:
                    if 'jpeg' in fourth or 'jpg' in fourth:
                        readfile = os.path.join(fourthpath, fourth)
                        writefile = os.path.join(savepath, fourth)
                        shutil.copy(readfile, writefile)


if __name__ == '__main__':
    path = '/VisualGroup/share/wujl/83/test/data'
    daypath = '/VisualGroup/share/wujl/83/test/days'
    os.makedirs(daypath, exist_ok=True)
    names = os.listdir(path)
    for name in names:
        picpath = os.path.join(path, name)
        savepath = os.path.join(daypath, name)
        os.makedirs(savepath, exist_ok=True)
        print(savepath)
        cp(picpath, savepath)
