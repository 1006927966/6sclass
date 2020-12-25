import os
import shutil

maplist = ['badcase', 'bancai_luan', 'bancai_zq', 'feigang', 'hanzha_zq', 'hanzha_l',
           'zhixiang_luan', 'zhixiang_zhenqgi', 'laji00', 'laji01', 'laji02', 'laji03',
           'luanfang', 'pip_l', 'pip_mid', 'pip_zq', 'xiaojian_luan', 'xiaojian_zhengqi',
           'xianshu_luan', 'xianshu_zhengqi', 'zangwu', 'zhufei']

fglist = ['bancai_luan', 'hanzha_l', 'zhixiang_luan', 'zhixiang_zhenqgi', 'laji00', 'laji01', 'laji02',
          'laji03', 'luanfang', 'pip_l', 'xiaojian_luan', 'xianshu_luan', 'zangwu', 'zhufei', 'feigang']

def jugeclassi(key):
    key = key[0]
    if '物料乱放' in key:
        return '物料乱放'
    elif '物品乱放' in key:
        return '物品乱放'
    elif '物料混放' in key:
        return '物料乱放'
    elif '物料占道' in key:
        return '物料占道'
    elif '物料摆放不整齐' in key:
        return '物料摆放不整齐'
    elif '垃圾' in key:
        return '垃圾'
    elif '地面脏' in key:
        return '地面脏'
    else:
        return 'others'


def calrecall(project, recallname, epoch, save):
    picspath = '/defaultShare/share/wujl/83/test/{}/croppic/{}'.format(project, recallname)
    origintxt = '/defaultShare/share/wujl/83/test/{}/txtpath/{}.txt'.format(project, recallname)
    pretxt = '/defaultShare/share/wujl/83/test/{}/txtpath/{}_{}.txt'.format(project, recallname, epoch)
    binarydir = '/defaultShare/share/wujl/83/test/{}/save/binary'.format(project)
    os.makedirs(binarydir, exist_ok=True)
    classidir = '/defaultShare/share/wujl/83/test/{}/save/classi'.format(project)
    os.makedirs(classidir, exist_ok=True)
    with open(pretxt, 'r') as f:
        prelines = f.readlines()
    with open(origintxt, 'r', encoding='utf-8') as f:
        originlines = f.readlines()
    picnames = os.listdir(picspath)
    predic = {}
    origindic = {}
    for preline in prelines:
        preline = preline.strip()
        prefactor = preline.split(',')
        predic[prefactor[0]] = prefactor[1:]
    for originline in originlines:
        originline = originline.strip()
        originfactor = originline.split(',')
        origindic[originfactor[0]] = originfactor[1:]
    binarydic = {}
    classidic = {}
    for picname in picnames:
        classname = jugeclassi(origindic[picname])
        preresult = predic[picname]
        clslabel = int(preresult[1])
        binarylabel = int(preresult[-1])
        if save:
            classipath = os.path.join(classidir, maplist[clslabel])
            os.makedirs(classipath, exist_ok=True)
            binarypath = os.path.join(binarydir, str(binarylabel))
            os.makedirs(binarypath, exist_ok=True)
            readfile = os.path.join(picspath, picname)
            writefile2 = os.path.join(binarypath, picname)
            writefilen = os.path.join(classipath, picname)
            shutil.copy(readfile, writefile2)
            shutil.copy(readfile, writefilen)
        if binarylabel == 1:
            if classname not in binarydic.keys():
                binarydic[classname] = 1
            else:
                binarydic[classname] += 1
        if maplist[clslabel] in fglist:
            if classname not in classidic.keys():
                classidic[classname] = 1
            else:
                classidic[classname] += 1
    for key in binarydic.keys():
        print('binary')
        print('[*]! {} is :{}'.format(key, binarydic[key]))
    print('*'*10)
    for key in classidic.keys():
        print('[*]! {} is :{}'.format(key, classidic[key]))


def calpre(project, prename, epoch):
    txtfile = '/defaultShare/share/wujl/83/test/{}/txtpath/{}_{}.txt'.format(project, prename, epoch)
    with open(txtfile, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(',') for line in lines]
    bdic = {}
    cdic = {}
    for line in lines:
        blabel = int(line[-1])
        clabel = maplist[int(line[-3])]
        if blabel not in bdic.keys():
            bdic[blabel] = 1
        else:
            bdic[blabel] += 1
        if clabel not in cdic.keys():
            cdic[clabel] = 1
        else:
            cdic[clabel] += 1
    for key in bdic.keys():
        print('binary')
        print('[*]! {} is :{}'.format(key, bdic[key]))
    print('*'*10)
    for key in cdic.keys():
        print('[*]! {} is :{}'.format(key, cdic[key]))



if __name__ == '__main__':
    project = '9_21'
    epoch = 55
    save = True
    #recallname = 'croprecall_0.6_0.01_0.1'
    prename = 'badcase'
    #calrecall(project, recallname, epoch, save)
    calpre(project, prename, epoch)