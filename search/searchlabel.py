import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
import os
import sys
path = os.path.abspath(os.path.dirname(__file__))
path = os.path.split(path)[0]
sys.path.append(path)
from models.resnext import make_model
import torch.nn as nn
from torchvision import models

class ListLoader(data.Dataset):
    def __init__(self, path_list):
        self.path_list = path_list
        self.input_h = 224
        self.input_w = 224
        self.mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(*self.mean_std)
        ])

    def _pre_process(self, raw_img):
        raw_img = Image.fromarray(raw_img)
        raw_img = self.transform(raw_img)
        return raw_img

    def __getitem__(self, idx):
        m_img = cv2.imread(self.path_list[idx])
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
        m_img = cv2.resize(m_img, (224, 224), cv2.INTER_LINEAR)
        m_img = self._pre_process(m_img)
        return m_img

    def __len__(self):
        return len(self.path_list)




class FeatureSearchAPI:
    def __init__(self, modelpath, npzpath):
        self.modelpath = modelpath
        self.npzpath = npzpath
        self.posmatrix, self.negmatrix = self.getmatrix()
        self.model = self.loadmodel()

    def loadmodel(self):
        net = models.__dict__['resnext101_32x8d'](pretrained=True)
        #net = make_model('resnext101_32x8d')
        #statedict = torch.load(self.modelpath, map_location='cpu')
        #net.load_state_dict(statedict)
        net = nn.Sequential(*list(net.children())[:-1])
        if torch.cuda.is_available():
            net.cuda(0)
        net.eval()
        return net

    def getmatrix(self):
        pospath = os.path.join(self.npzpath, 'posim.npz')
        negpath = os.path.join(self.npzpath, 'negim.npz')
        posdata = np.load(pospath)
        negdata = np.load(negpath)
        self.posmatrix = posdata['arr_0']
        self.negmatrix = negdata['arr_0']
        return self.posmatrix, self.negmatrix


    def splitjudge(self, posindexs, negindexs, posdistance, negdistance):
        posnum = 0
        negnum = 0
        for posindex in posindexs:
            posnum += posdistance[posindex]
        for negindex in negindexs:
            negnum += negdistance[negindex]
        if posnum >= negnum:
            return 1
        else:
            return 0

    def alljudge(self, posnum, allindexs):
        pos = 0
        neg = 0
        for allindex in allindexs:
            if allindex >= posnum:
                neg += 1
            else:
                pos += 1
        if pos >= neg:
            return 1
        else:
            return 0

# get the dict {split:splitlabel, all:alllabel}
# the label in ranked by ranks
    def gettwolabel(self, ranks, arr1s):
        imgnum = arr1s.shape[0]
        posnum = self.posmatrix.shape[0]
        dic = {}
        posdistance = np.matmul(arr1s, self.posmatrix.T)

        negdistance = np.matmul(arr1s, self.negmatrix.T)
        print(posdistance.shape)
        print(negdistance.shape)
        alldistance = np.hstack((posdistance, negdistance))
        posindexs = np.argsort(-posdistance, 1)
        negindexs = np.argsort(-negdistance, 1)
        allindexs = np.argsort(-alldistance, 1)
        for i in range(imgnum):
            splitlabels = []
            alllabels = []
            negdistance1 = negdistance[i]
            posdistance1 = posdistance[i]
            allindexs1 = allindexs[i]
            negindexs1 = negindexs[i]
            posindexs1 = posindexs[i]
            for rank in ranks:
                posindex = posindexs1[:rank]
                negindex = negindexs1[:rank]
                allindex = allindexs1[:rank]
                splitlabel = self.splitjudge(posindex, negindex, posdistance1, negdistance1)
                alllabel = self.alljudge(posnum, allindex)
                splitlabels.append(splitlabel)
                alllabels.append(alllabel)
            dic[i] = {'split':splitlabels, 'all':alllabels}
        return dic

    def infer(self, x, ranks):
        with torch.no_grad():
            y = self.model(x)
            y = y.view(y.size(0), -1)
            y = nn.functional.normalize(y, 1)
            y = y.cpu().numpy()
            redic = self.gettwolabel(ranks, y)
        return redic




class MOCOSearchAPI:
    def __init__(self, modelpath, npzpath):
        self.modelpath = modelpath
        self.npzpath = npzpath
        self.posmatrix, self.negmatrix = self.getmatrix()
        self.model = self.loadmodel()

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

    def getmatrix(self):
        pospath = os.path.join(self.npzpath, 'mocopos.npz')
        negpath = os.path.join(self.npzpath, 'moconeg.npz')
        posdata = np.load(pospath)
        negdata = np.load(negpath)
        self.posmatrix = posdata['arr_0']
        self.negmatrix = negdata['arr_0']
        return self.posmatrix, self.negmatrix


    def splitjudge(self, posindexs, negindexs, posdistance, negdistance):
        posnum = 0
        negnum = 0
        for posindex in posindexs:
            posnum += posdistance[posindex]
        for negindex in negindexs:
            negnum += negdistance[negindex]
        if posnum >= negnum:
            return 1
        else:
            return 0

    def alljudge(self, posnum, allindexs):
        pos = 0
        neg = 0
        for allindex in allindexs:
            if allindex >= posnum:
                neg += 1
            else:
                pos += 1
        if pos >= neg:
            return 1
        else:
            return 0

# get the dict {split:splitlabel, all:alllabel}
# the label in ranked by ranks
    def gettwolabel(self, ranks, arr1s):
        imgnum = arr1s.shape[0]
        posnum = self.posmatrix.shape[0]
        dic = {}
        posdistance = np.matmul(arr1s, self.posmatrix.T)

        negdistance = np.matmul(arr1s, self.negmatrix.T)
        print(posdistance.shape)
        print(negdistance.shape)
        alldistance = np.hstack((posdistance, negdistance))
        posindexs = np.argsort(-posdistance, 1)
        negindexs = np.argsort(-negdistance, 1)
        allindexs = np.argsort(-alldistance, 1)
        for i in range(imgnum):
            splitlabels = []
            alllabels = []
            negdistance1 = negdistance[i]
            posdistance1 = posdistance[i]
            allindexs1 = allindexs[i]
            negindexs1 = negindexs[i]
            posindexs1 = posindexs[i]
            for rank in ranks:
                posindex = posindexs1[:rank]
                negindex = negindexs1[:rank]
                allindex = allindexs1[:rank]
                splitlabel = self.splitjudge(posindex, negindex, posdistance1, negdistance1)
                alllabel = self.alljudge(posnum, allindex)
                splitlabels.append(splitlabel)
                alllabels.append(alllabel)
            dic[i] = {'split':splitlabels, 'all':alllabels}
        return dic

    def infer(self, x, ranks):
        with torch.no_grad():
            y = self.model(x)
            y = nn.functional.normalize(y, 1)
            y = y.cpu().numpy()
            redic = self.gettwolabel(ranks, y)
        return redic


def tester(imgpath, modelpath, npzpath, ranks, tag):
    names = os.listdir(imgpath)
    pathlist = [os.path.join(imgpath, name) for name in names]
    dataset = ListLoader(pathlist)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=False)
    if tag == 'feature':
        searchApi = FeatureSearchAPI(modelpath, npzpath)
        txtpath1 = os.path.join(os.path.dirname(imgpath), 'imfeatures.txt')
        txtpath2 = os.path.join(os.path.dirname(imgpath), 'imfeaturea.txt')
    else:
        searchApi = MOCOSearchAPI(modelpath, npzpath)
        txtpath1 = os.path.join(os.path.dirname(imgpath), 'mocos.txt')
        txtpath2 = os.path.join(os.path.dirname(imgpath), 'mocoa.txt')
    for index, image in enumerate(dataloader):
        image = image.cuda()
        print(index*image.size(0)/len(names))
        print('*'*5)
        redic = searchApi.infer(image, ranks)
        subnames = names[index * image.size(0): (index + 1) * image.size(0)]
        with open(txtpath1, 'a') as f:
            for key in redic.keys():
                splitvalues = redic[key]['split']
                f.write('{}, {}, {}, {}, {}, {}, {}\n'.format(subnames[key], splitvalues[0], splitvalues[1], splitvalues[2],
                                                              splitvalues[3], splitvalues[4], splitvalues[5]))
        with open(txtpath2, 'a') as f:
            for key in redic.keys():
                splitvalues = redic[key]['all']
                f.write('{}, {}, {}, {}, {}, {}, {}\n'.format(subnames[key], splitvalues[0], splitvalues[1], splitvalues[2],
                                                          splitvalues[3], splitvalues[4], splitvalues[5]))


if __name__ == '__main__':
    imgpath = '/VisualGroup/share/wujl/83/test/9_21/croppic/crop_0.1/badcase'
    print(imgpath)
    #mocomodelpath = '/defaultShare/share/wujl/moco/featuremodels/checkpoint_0199.pth.tar'
    fetmodelpath = '/defaultShare/share/wujl/83/master_models/resnext101_22/model_50.pth'
    npzpath = '/VisualGroup/share/wujl/83/search/npzdir'
    ranks = [5, 10, 20, 40, 50, 100]
    tester(imgpath, fetmodelpath, npzpath, ranks, 'feature')
    #tester(imgpath, mocomodelpath, npzpath, ranks, 'moco')

