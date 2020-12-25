import os
import shutil

def cal(txtpath, maplist, thresh):
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    sum = 0
    count = 0
    for line in lines:
        sum += 1
        line = line.strip()
        factors = line.split(',')
        label = int(factors[-3])
        score = float(factors[-4])
        if label in maplist and score>=thresh:
            count += 1
    return count, sum, count/sum


def calm(txtpath, thresh):
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    sum = 0
    count = 0
    for line in lines:
        sum += 1
        line = line.strip()
        factors = line.split(',')
        score = float(factors[-2])
        if score > thresh:
            count += 1
    return count, sum, count/sum


def displayhard(txtpath, origindir, savedir, maplist, labelnames):
    count = 0
    os.makedirs(savedir, exist_ok=True)
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        count += 1
        line = line.strip()
        factors = line.split(',')
        label = int(factors[-1])
        name = factors[0]
        print(name)
        if label in maplist:
            labelname = labelnames[label]
            readfile = os.path.join(origindir, name)
            writefile = os.path.join(savedir, labelname+'{}.jpg'.format(count))
            shutil.copy(readfile, writefile)






if __name__ == '__main__':
    names =  ['badcase', 'bancai_luan', 'bancai_zq', 'feigang', 'hanzha_zq', 'hanzha_l',
            'zhixiang_luan', 'zhixiang_zhenqgi', 'laji00', 'laji01', 'laji02', 'laji03',
            'luanfang', 'pip_l', 'pip_mid', 'pip_zq', 'xiaojian_luan', 'xiaojian_zhengqi',
            'xianshu_luan', 'xianshu_zhengqi', 'zangwu', 'zhufei', 'people', 'gulixj', 'cr']
    txtpath = '/VisualGroup/share/wujl/83/test/testcls/txtpath/bg_test_60_bcnn.txt'
    #maplist = [1, 3, 5, 6, 8, 12, 13, 14, 16, 18]
    #clsscores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    maplist = [1, 5, 6, 8, 10, 11, 12, 13, 14, 16, 18, 20, 21, 24]
    origindir = '/VisualGroup/share/wujl/83/test/testcls/bg_test'
    savedir = '/VisualGroup/share/wujl/83/test/testcls/badcase/bg_bcnn'
    displayhard(txtpath, origindir, savedir, maplist, names)