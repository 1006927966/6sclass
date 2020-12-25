import os
import shutil

def combinedata(root, day, picsdir, xmsdir):
    srcdir = os.path.join(root, day)
    anno = os.path.join(srcdir, 'Annotations')
    jpeg = os.path.join(srcdir, 'JPEGImages')
    imgnames = os.listdir(jpeg)
    for imgname in imgnames:
        ip = imgname[:-5]
        print(ip)
        xmlname = ip+'.xml'
        xmlreadfile = os.path.join(anno, xmlname)
        imgreadfile = os.path.join(jpeg, imgname)
        xmlwritefile = os.path.join(xmsdir, xmlname)
        imgwritefile = os.path.join(picsdir, imgname)
        shutil.copy(xmlreadfile, xmlwritefile)
        shutil.copy(imgreadfile, imgwritefile)

if __name__ == '__main__':
    root = '/defaultShare/share/wujl/83/online_data/faildata'
    days = os.listdir(root)
    picsdir = '/defaultShare/share/wujl/83/online_data/recalldir/JPEGImages'
    xmsdir = '/defaultShare/share/wujl/83/online_data/recalldir/Annotations'
    os.makedirs(picsdir, exist_ok=True)
    os.makedirs(xmsdir, exist_ok=True)
    for day in days:
        if '2020-10' not in day:
            print(day)
            combinedata(root, day, picsdir, xmsdir)
