import cv2
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader
import os
import shutil
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        print(self.path_list[idx])
        m_img = cv2.imread(self.path_list[idx])
        print(m_img)
        print('*')
        m_img = cv2.cvtColor(m_img, cv2.COLOR_BGR2RGB)
        m_img = cv2.resize(m_img, (224, 224), cv2.INTER_LINEAR)
        m_img = self._pre_process(m_img)
        return m_img

    def __len__(self):
        return len(self.path_list)


class ClassifierApi(object):
    def __init__(self, **kwargs):

        self.model_path = kwargs.get('model_path') or 'frozen_model.pt'
        self.device = kwargs.get('device') or None
        self.n_gpu = kwargs.get('n_gpu') or None
        self.gpu_list = list(range(self.n_gpu))


        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._construct_model(self.model_path)

        pass

    def _construct_model(self, model_path):
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model

    def _infer(self, x):
        with torch.no_grad():
            if self.n_gpu > 1:
                y = torch.nn.parallel.data_parallel(self.model, x, self.gpu_list)
            else:
                y = self.model(x)

            y = torch.nn.functional.softmax(y, dim=1)
            value, label = torch.max(y, 1)
            label = label.cpu().numpy()
            value = value.cpu().numpy()
            return value, label

    def exec(self, tensor):
        value, label = self._infer(tensor.to(self.device))
        return dict(score=[value, label])


def test(modelpath, dirpath):
    txtpath = dirpath+ 'txt'
    os.makedirs(txtpath, exist_ok=True)
    my_inference = ClassifierApi(model_path=modelpath, n_gpu=1)
    img_dirs = os.listdir(dirpath)
    for img_dir in img_dirs:
        save_txt = os.path.join(txtpath, '{}.txt'.format(img_dir))
        txt_file = open(save_txt, 'w')
        imgpath = os.path.join(dirpath, img_dir)
        name_list = os.listdir(imgpath)
        img_list = list()
        for name in name_list:
            img_list.append(os.path.join(imgpath, name))
        data_set = ListLoader(path_list=img_list)
        data_loader = DataLoader(data_set, batch_size=16, num_workers=4, shuffle=False)
        score_list = list()
        label_list = list()
        for index, img in enumerate(data_loader):
            batch_score = my_inference.exec(img)
            print(batch_score)
            scores, label = batch_score['score']
            score_list.extend(list(scores))
            label_list.extend(list(label))
            print(score_list)
            print('{}/{}'.format((index + 1) *len(list(batch_score['score'])), len(name_list)))
        print(np.sum(score_list))
        for index, name in enumerate(name_list):
            txt_file.write('{}, {}, {}\n'.format(name, score_list[index], label_list[index]))
        txt_file.close()

def calculate(txtdir, mapindexs, thresh):
    saveroot = txtdir[:-3] + 'badcase'
    os.makedirs(saveroot, exist_ok=True)
    dirroot = txtdir[:-3]
    cdic = {}
    txtnames = os.listdir(txtdir)
    for txtname in txtnames:
        classiname = txtname.split('.')[0]
        print(classiname)
        if classiname == 'wlzd':
            continue
        imgdirpath = os.path.join(dirroot, classiname)
        txtpath = os.path.join(txtdir, txtname)
        with open(txtpath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            factors = line.split(',')
            score = float(factors[1])
            label = int(factors[2])
            imgname = factors[0]
            if (score > thresh) and (label in mapindexs):
                if classiname not in cdic.keys():
                    cdic[classiname] = 1
                else:
                    cdic[classiname] += 1
            else:
                imgorignname = os.path.join(imgdirpath, imgname)
                writefile = os.path.join(saveroot, imgname)
                shutil.copy(imgorignname, writefile)
    cdic['wlzd'] = 0
    return cdic

def parse_dic(predic, recalldic, excelpath):
    with open(excelpath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            if i == 2:
                detprerow = rows
            if i == 3:
                detrecalrow = rows
    order = ['wplf', 'wllf', 'wlzd', 'wlbfbzq', 'lj', 'dmz', 'others']
    recalrow = ['recall']
    prerow = ['pre']
    precount = 0
    recallcount = 0
    for i in range(len(order)):
        key = order[i]
        if key == 'wlzd':
            precount += 0
            recallcount += 0
        precount += predic[key]
        recallcount += recalldic[key]
        prerow.append(str(predic[key]))
        recalrow.append(str(recalldic[key]))
    prerow.append(str(precount))
    recalrow.append(str(recallcount))
    print(prerow)
    print(recalrow)
    print(detprerow)
    print(detrecalrow)
    rateprerow = ['prerate']
    raterecalrow = ['recalrate']
    for i in range(1, len(prerow)):
        print(int(prerow[i])/int(detprerow[i]))
        rateprerow.append(str(int(prerow[i])/int(detprerow[i])))
        raterecalrow.append(str(int(recalrow[i])/int(detrecalrow[i])))
    with open(excelpath, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(prerow)
        writer.writerow(recalrow)
        writer.writerow(rateprerow)
        writer.writerow(raterecalrow)


def deletetxt(txtpath):
    names = os.listdir(txtpath)
    for name in names:
        path = os.path.join(txtpath, name)
        os.remove(path)
    os.removedirs(txtpath)

def classimain(modelpath, predirpath, recalldirpath, excelpath, mapindexs, thresh):
    test(modelpath, predirpath)
    test(modelpath, recalldirpath)
    pretxtpath = predirpath+ 'txt'
    recalltxtpath = recalldirpath + 'txt'
    recalldic = calculate(recalltxtpath, mapindexs, thresh)
    predic = calculate(pretxtpath, mapindexs, thresh)
    parse_dic(predic, recalldic, excelpath)
    deletetxt(pretxtpath)
    deletetxt(recalltxtpath)


if __name__ == '__main__':
    modelpath = '/defaultShare/share/wujl/83/master_models/resnext101_19mixcut/model_58_frozen.pt'
    predirpath = '/defaultShare/share/wujl/83/test/tt23/cropDet'
    recalldirpath = '/defaultShare/share/wujl/83/test/tt23/cropRecall'
    excelpath = '/defaultShare/share/wujl/83/test/tt23/a.csv'
    mapindexs = [3, 4, 7, 10, 12, 14, 17]
    thresh = 0
    classimain(modelpath, predirpath, recalldirpath, excelpath, mapindexs, thresh)


