import os
import shutil

def getlabeldic(odic, txtpath):
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        factors = line.split(',')
        name = factors[0]
        label = factors[-1]
        if name not in odic.keys():
            odic[name] = [label]
        else:
            odic[name].append(label)
    return odic

def getesmblelabel(labelist):
    labeldic = {}
    for label in labelist:
        if label not in labeldic.keys():
            labeldic[label] = 1
        else:
            labeldic[label] += 1
    amax = 0
    label = ''
    for key in labeldic.keys():
        if labeldic[key] > amax:
            amax = labeldic[key]
            label = key
    return label

def savebylabel(odic, origindir, savepath, maplist):
    for key in odic.keys():
        labelist = odic[key]
        label = getesmblelabel(labelist)
        label = maplist[int(label)]
        print(label)
        savefile = os.path.join(savepath, label)
        os.makedirs(savefile, exist_ok=True)
        writefile = os.path.join(savefile, key)
        readfile = os.path.join(origindir, key)
        shutil.copy(readfile, writefile)


if __name__ == '__main__':
    maplist = ['badcase', 'feigang', 'hanzha', 'laji00', 'laji01', 'laji02', 'laji03', 'luanfang', 'zangwu',
           'zhufei', 'xianshu_luan', 'xianshu_zhengqi', 'xiaojian_luan', 'xiaojian_zhengqi', 'pip_luan', 'pip_zhengqi', 'zhixiang',
               'bancai_luan', 'bancai_zhenqgi']
    pads = [0.1, 0.2, 0.3, 0.4, 0.5]
    savepredir = '/defaultShare/share/wujl/83/test/9_21/pre_{}'.format(len(pads))
    saverecalldir = '/defaultShare/share/wujl/83/test/9_21/recall_{}'.format(len(pads))
    os.makedirs(savepredir, exist_ok=True)
    os.makedirs(saverecalldir, exist_ok=True)
    originpredir = '/defaultShare/share/wujl/83/test/9_21/croppic/croppre_0.6_0.1'
    originrecalldir = '/defaultShare/share/wujl/83/test/9_21/croppic/croprecall_0.6_0.3_0.1'
    predic = {}
    recalldic = {}
    for pad in pads:
        txtprepath = '/defaultShare/share/wujl/83/test/9_21/txtpath/croppre_0.6_{}.txt'.format(pad)
        txtrecallpath = '/defaultShare/share/wujl/83/test/9_21/txtpath/croprecall_0.6_0.3_{}.txt'.format(pad)
        print(txtprepath)
        predic = getlabeldic(predic, txtprepath)
        recalldic = getlabeldic(recalldic, txtrecallpath)
    savebylabel(predic, originpredir, savepredir, maplist)
    savebylabel(recalldic, originrecalldir, saverecalldir, maplist)


