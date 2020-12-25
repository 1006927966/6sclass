import numpy as np


def parsedbtxt(txtpath):
    dic = {}
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        name = line.split(',')[0]
        label = int(line.split(',')[-1])
        dic[name] = label
    return dic


def vottingdb(pads, project, epoch):
    if len(pads) == 5:
        num = 3
    else:
        num = 2
    diclist = []
    for pad in pads:
        txtpath = '/defaultShare/share/wujl/83/test/{}/txtpath/crop_{}_{}.txt'.format(project, pad, epoch)
        diclist.append(parsedbtxt(txtpath))
    sum = 0
    count = 0
    for key in diclist[0].keys():
        sum += 1
        labellist = []
        for dic in diclist:
            labellist.append(dic[key])
        labelsum = np.sum(np.array(labellist))
        if labelsum >= 3:
            count += 1
    return count/sum, sum

if __name__ == '__main__':
    padslist = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4, 0.5], [0.3, 0.4, 0.5]]
    projects = ['9_21', 'bg_test']
    epoch = 55
    for project in projects:
        for pads in padslist:
            rate, count = vottingdb(pads, project, epoch)
            print(project)
            print(pads)
            print(rate)
            print('[*]'*10)


