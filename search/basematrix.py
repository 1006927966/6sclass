import torch

import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os
import sys
path = os.path.abspath(os.path.dirname(__file__))
path = os.path.split(path)[0]
sys.path.append(path)
from models.resnext import make_model
import numpy as np
from torchvision import models


class BASE:
    def __init__(self, modelpath, savedir, origndir):
        self.modelpath = modelpath
        self.savedir = savedir
        self.origndir = origndir

    def loadmodel(self):
        pass

    def infer(self):
        pass

    def savenp(self):
        pass


class Feature(BASE):
    def __init__(self, modelpath, savedir, origndir):
        super(Feature, self).__init__(modelpath, savedir, origndir)
        self.model = self.loadmodel()

    def getinput(self, path):
        img = Image.open(path)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        return img

    def loadmodel(self):
        net = models.__dict__['resnext101_32x8d'](pretrained=True)
       # net = make_model('resnext101_32x8d')
       # statedict = torch.load(self.modelpath, map_location='cpu')
       # net.load_state_dict(statedict)
        net = nn.Sequential(*list(net.children())[:-1])
        if torch.cuda.is_available():
            net.cuda(0)
        net.eval()
        return net

    def infer(self):
        posmap = [1, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 21]
        neglist = []
        poslist = []
        names = os.listdir(self.origndir)
        for name in names:
            classdir = os.path.join(self.origndir, name)
            picnames = os.listdir(classdir)
            sum = 0
            for picname in picnames:
                sum += 1
                if sum >= 500:
                    break
                picpath = os.path.join(classdir, picname)
                try:
                    img = self.getinput(picpath)
                except:
                    continue
                print(picpath)
                img = img.cuda()
                net = self.loadmodel()
                with torch.no_grad():
                    label = net(img)
                    label = label.view(label.size(0), -1)
                    label = nn.functional.normalize(label, 1)
                label = label.cpu().numpy()
                if int(name) in posmap:
                    poslist.append(label[0])
                else:
                    neglist.append(label[0])
        return np.array(poslist), np.array(neglist)

    def savenp(self):
        posarr, negarr = self.infer()
        np.savez(os.path.join(self.savedir, 'posim'), posarr)
        print('[*]! pis is save!!!!!')
        np.savez(os.path.join(self.savedir, 'negim'), negarr)
        print('[*]! neg is save!!!!!')


class MOCO(BASE):
    def __init__(self, modelpath, savedir, origndir):
        super(MOCO, self).__init__(modelpath, savedir, origndir)
        self.model = self.loadmodel()

    def getinput(self, path):
        img = Image.open(path)
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        return img

    def loadmodel(self):
        model = models.__dict__['resnet50'](num_classes=128)
        dics = torch.load(self.modelpath, map_location='cpu')
        state_dict = dics['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model

    def infer(self):
        posmap = [1, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 21]
        neglist = []
        poslist = []
        names = os.listdir(self.origndir)
        for name in names:
            classdir = os.path.join(self.origndir, name)
            picnames = os.listdir(classdir)
            sum = 0
            for picname in picnames:
                sum += 1
                if sum >= 500:
                    break
                picpath = os.path.join(classdir, picname)
                try:
                    img = self.getinput(picpath)
                except:
                    continue
                print(picpath)
                img = img.cuda()
                net = self.loadmodel()
                with torch.no_grad():
                    label = net(img)
                    label = nn.functional.normalize(label, 1)
                label = label.cpu().numpy()
                if int(name) in posmap:
                    poslist.append(label[0])
                else:
                    neglist.append(label[0])
        return np.array(poslist), np.array(neglist)

    def savenp(self):
        posarr, negarr = self.infer()
        np.savez(os.path.join(self.savedir, 'mocopos'), posarr)
        print('[*]! pis is save!!!!!')
        np.savez(os.path.join(self.savedir, 'moconeg'), negarr)
        print('[*]! neg is save!!!!!')


if __name__ == '__main__':
    respath = '/defaultShare/share/wujl/83/master_models/resnext101_22/model_50.pth'
    mocopath = '/defaultShare/share/wujl/moco/featuremodels/checkpoint_0199.pth.tar'
    origndir = '/VisualGroup/share/wujl/83/ddb_22/train'
    savedir = '/VisualGroup/share/wujl/83/search/npzdir'
    feature = Feature(respath, savedir, origndir)
    feature.savenp()
    #moco = MOCO(mocopath, savedir, origndir)
    #moco.savenp()


