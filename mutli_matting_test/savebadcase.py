import os
import shutil

def savebad(project, pad, tag, epoch):
    imgdir = '/defaultShare/share/wujl/83/test/{}/croppic/crop_{}/badcase'.format(project, pad)
    txtpath = '/defaultShare/share/wujl/83/test/{}/txtpath/crop_{}_{}.txt'.format(project, pad, epoch)
    save = '/defaultShare/share/wujl/83/test/{}/badcase'.format(project)
    os.makedirs(save, exist_ok=True)
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    print(txtpath)
    for line in lines:
        line = line.strip()
        factors = line.split(',')
        name = factors[0]
        label = int(factors[-1])
        if label == tag:
            print(name)
            writepath = os.path.join(save, name)
            readpath = os.path.join(imgdir, name)
            shutil.copy(readpath, writepath)

if __name__ == '__main__':
    projects = ['9_21', 'bg_test']
    pad = 0.1
    epoch = 55
    thresh = 0.5
    savebad('9_21', pad, 0, epoch)
    savebad('bg_test', pad, 1, epoch)
