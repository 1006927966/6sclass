import os
import shutil

def cp(imgdir, xmldir, savedir):
    xmlnames = os.listdir(xmldir)
    for xmlname in xmlnames:
        imgname = xmlname[:-4] + '.jpg'
        print(imgname)
        imgpath = os.path.join(imgdir, imgname)
        savepath = os.path.join(savedir, imgname)
        shutil.copy(imgpath, savepath)


if __name__ == '__main__':
    imgdir = '/defaultShare/share/wujl/83/test/bg_test/JPEGImages'
    xmldir = '/defaultShare/share/wujl/83/test/bg_test/det_xml'
    savedir = '/defaultShare/share/wujl/83/test/bg_test/det_img'
    os.makedirs(savedir, exist_ok=True)
    cp(imgdir, xmldir, savedir)