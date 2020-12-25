import os

def calnum(txtpath, num):
    with open(txtpath, 'r') as f:
        lines = f.readlines()
    all = len(lines)
    cal = [0]*num
    for line in lines:
        line = line.strip()
        factors = line.split(',')
        for i in range(num):
            if int(factors[i+1]) == 1:
                cal[i] += 1
    return [a/all for a in cal]


if __name__ == '__main__':
    rootpath = '/VisualGroup/share/wujl/83/test/9_21/croppic/crop_0.1'
    names = ['imfeaturea.txt', 'imfeatures.txt']
    for name in names:
        print(name)
        txtpath = os.path.join(rootpath, name)
        rate = calnum(txtpath, 6)
        print(rate)
        print('*'*6)