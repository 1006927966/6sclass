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

from utils.model_select import get_network
from config.value_config import *
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
        m_img = cv2.imread(self.path_list[idx])
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
        model = get_network('multilabel', NUMCLASS)
        statedic = torch.load(model_path, map_location='cpu')
        model.load_state_dict(statedic)
        model.to(self.device)
        model.eval()
        return model

    def _infer(self, x):
        with torch.no_grad():
            if self.n_gpu > 1:
                y = torch.nn.parallel.data_parallel(self.model, x, self.gpu_list)
            else:
                y1, y2 = self.model(x)
            y1 = torch.nn.functional.softmax(y1, dim=1)
            value1, label1 = torch.max(y1, 1)
            label1 = label1.cpu().numpy()
            value1 = value1.cpu().numpy()

            y2 = torch.nn.functional.softmax(y2, dim=1)
            value2, label2 = torch.max(y2, 1)
            label2 = label2.cpu().numpy()
            value2 = value2.cpu().numpy()

            return value1, label1, value2, label2

    def exec(self, tensor):
        value1, label1, value2, label2 = self._infer(tensor.to(self.device))
        return dict(score=[value1, label1, value2, label2])


def test(epoch, project, dirname):
    print('1')
    my_inference = ClassifierApi(model_path='/VisualGroup/share/wujl/83/master_models/resnext101_25/model_{}.pth'.format(epoch), n_gpu=1)
    print('2')
    img_dir = '/VisualGroup/share/wujl/83/test/{}/{}/'.format(project, dirname)
    txt_dir = '/VisualGroup/share/wujl/83/test/{}/txtpath'.format(project)
    os.makedirs(txt_dir, exist_ok=True)
    print(img_dir)
    save_txt = os.path.join(txt_dir, '{}_{}_25.txt'.format(dirname, epoch))
    print(save_txt)
    txt_file = open(save_txt, 'w')
    name_list = os.listdir(img_dir)
    img_list = list()
    for name in name_list:
        img_list.append(os.path.join(img_dir, name))
    data_set = ListLoader(path_list=img_list)
    data_loader = DataLoader(data_set, batch_size=2, num_workers=4, shuffle=False)
    value1_list = list()
    label1_list = list()
    value2_list = list()
    label2_list = list()
    for index, img in enumerate(data_loader):
        batch_score = my_inference.exec(img)
        print(batch_score)
        value1, label1, value2, label2 = batch_score['score']
        value1_list.extend(list(value1))
        label1_list.extend(list(label1))
        value2_list.extend(list(value2))
        label2_list.extend(list(label2))
        print('{}/{}'.format((index + 1) *len(list(batch_score['score'])), len(name_list)))
    print(np.sum(value1_list))
    for index, name in enumerate(name_list):
        txt_file.write('{}, {}, {}, {}, {}\n'.format(name, value1_list[index], label1_list[index], value2_list[index], label2_list[index]))
    txt_file.close()

if __name__ == '__main__':
    epochs = [65]
    project = 'testcls'
    dirnames = ['bg_test', 'recall_test']
    for epoch in epochs:
        for dirname in dirnames:
            print('[*]! the epoch {} begin'.format(epoch))
            test(epoch, project, dirname)
