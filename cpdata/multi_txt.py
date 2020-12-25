import os
import shutil

def dirpath(rootdir, name, mapindexs):
    imgroot = os.path.join(rootdir, name)
    txtroot = imgroot + 'txt'
    savedir = imgroot+'save'
    txtnames = os.listdir(txtroot)
    for txtname in txtnames:
        classiname = txtname.split('.')[0]
        savepath = os.path.join(savedir, classiname)
        originpath = os.path.join(imgroot, classiname)
        txtpath = os.path.join(txtroot, txtname)
        with open(txtpath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            imgname = line.split(',')[0]
            index = int(line.split(',')[-1])
            readfile = os.path.join(originpath, imgname)
            if index in  mapindexs:

                writedir = os.path.join(savepath, 'luan')
                os.makedirs(writedir, exist_ok=True)
                writefile = os.path.join(writedir, imgname)
                shutil.copy(readfile, writefile)
            else:
                writedir = os.path.join(savepath, 'zhengqi')
                os.makedirs(writedir, exist_ok=True)
                writefile = os.path.join(writedir, imgname)
                shutil.copy(readfile, writefile)

if __name__ == '__main__':
    rootdir = '/defaultShare/share/wujl/83/test/tt21'
    mapindexs = [2, 3, 5, 6, 7, 10, 14, 17]
    names = ['cropDet', 'cropRecall']
    for name in names:
        dirpath(rootdir, name, mapindexs)


