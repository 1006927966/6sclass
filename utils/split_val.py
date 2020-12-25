import shutil
import random
import os

def train_val(traindir, valdir):
    os.makedirs(valdir, exist_ok=True)
    names = os.listdir(traindir)
    n = len(names)
    valnum = n // 10
    for i in range(valnum):
        names = os.listdir(traindir)
        print(len(names))
        fac = random.choice(names)
        readfile = os.path.join(traindir, fac)
        writefile = os.path.join(valdir, fac)
        shutil.move(readfile, writefile)

if __name__ == '__main__':
    trainroot = '/VisualGroup/share/wujl/83/ddb_25/train'
    labels = os.listdir(trainroot)
    valroot = '/VisualGroup/share/wujl/83/ddb_25/val'
    for label in labels:
        print(label)
        traindir = os.path.join(trainroot, label)
        valdir = os.path.join(valroot, label)
        train_val(traindir, valdir)

