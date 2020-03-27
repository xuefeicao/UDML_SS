from __future__ import absolute_import, print_function
import torch
import torch.utils.data as data
from PIL import Image

import os
import sys
import numpy as np
from data import transforms 
from collections import defaultdict

Dict = {
    'cub': "CUB_200_2011",
    'car': "Cars196",
    'product': "Products",
}



def default_loader(path, self_supervision_rot=0):
    mat = np.array(Image.open(path).convert('RGB'))
    #mat = np.rot90(mat, np.random.choice(range(4)), axes=(0, 1))
    all_mats = [Image.fromarray(mat)]
    # test rotation influence

    if self_supervision_rot:
        for i in range(1, 4):
            all_mats.append(Image.fromarray(np.rot90(mat, i, axes=(0, 1))))
    return all_mats

def Generate_transform_Dict(origin_width=256, width=227, ratio=0.16, rot=0, args=None):
    
    std_value = 1.0 / 255.0
    if (args is not None) and ("ResNet" in args.net):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        cc = []
    else:
        normalize = transforms.Normalize(mean=[104 / 255.0, 117 / 255.0, 128 / 255.0],
                                     std= [1.0/255, 1.0/255, 1.0/255])
        print("bgr init")
        cc = [transforms.CovertBGR()]
    transform_dict = {}

    transform_dict['rand-crop'] = \
    transforms.Compose(cc + 
                [transforms.Resize((origin_width)),
                transforms.RandomResizedCrop(scale=(ratio, 1), size=width),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
               ])

    transform_dict['center-crop'] = \
    transforms.Compose(cc +
                [
                    transforms.Resize((origin_width)),
                    transforms.CenterCrop(width),
                    transforms.ToTensor(),
                    normalize,
                ])
    
    transform_dict['resize'] = \
    transforms.Compose(cc + 
                [    transforms.Resize((width)),
                    transforms.ToTensor(),
                    normalize,
                ])
    return transform_dict

def rot_tensor(ts):
    ts = ts[0].numpy()
    ans = [torch.FloatTensor(np.rot90(ts, i, axes=(1, 2)).copy()) for i in range(4)]
    return ans

        

class MyData(data.Dataset):
    def __init__(self, root=None, label_txt=None,
                 transform=None, loader=default_loader, self_supervision_rot=0, rot_bt=True, mode="train", corruption=0):

        # Initialization data path and train(gallery or query) txt path

        self.self_supervision_rot = self_supervision_rot
        self.rot_bt = rot_bt
        self.mode = mode
        print('transform used:', transform)
        
        if label_txt is None:
            raise Exception('wrong data used!')
            label_txt = os.path.join(root, 'train.txt')

        if transform is None:
            transform_dict = Generate_transform_Dict()['rand-crop']

        file = open(label_txt)
        images_anon = file.readlines()

        images = []
        labels = []

        for img_anon in images_anon:

            [img, label] = img_anon.split(' ')
            images.append(img)
            labels.append(int(label))

        classes = list(set(labels))



        # Generate Index Dictionary for every class
        Index = defaultdict(list)
        for i, label in enumerate(labels):
            Index[label].append(i)

        # Initialization Done
        self.root = root
        self.images = images
        self.labels = labels
        self.classes = classes
        self.transform = transform
        self.Index = Index
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.images[index], self.labels[index]
        if "result" not in self.root:
            fn = os.path.join(self.root, fn)
        if not self.rot_bt:
            all_mats = self.loader(fn, 0)
        else:
            all_mats = self.loader(fn, self.self_supervision_rot)
                    
        if self.transform is not None:
            for i, mat in enumerate(all_mats):
                all_mats[i] = self.transform(mat)
            if self.self_supervision_rot and not self.rot_bt:
                all_mats = rot_tensor(all_mats)
        
        img = torch.cat(all_mats, dim=0)
        
        if self.mode == "test":
            img = img[:3, :, :]
        return img, label

    def __len__(self):
        return len(self.images)


class Dataset:
    def __init__(self, whichdata, width=227, origin_width=256, ratio=0.16, root=None, transform=None, mode="train", self_supervision_rot=0, rot_bt=1, corruption=0, args=None):
        print('width: \t {}'.format(width))
        root = os.path.join(root, Dict[whichdata])
        transform_Dict = Generate_transform_Dict(origin_width=origin_width, width=width, ratio=ratio, args=args)
        
        train_txt = "train.txt"
        if corruption > 0:
            train_txt = "train_" + str(corruption) + ".txt"
        print(train_txt + " is used!")

        

        train_txt = os.path.join(root, train_txt)
        test_txt = os.path.join(root, 'test.txt')
       
        self.train = MyData(root, label_txt=train_txt, transform=transform_Dict['rand-crop'], mode=mode, self_supervision_rot=self_supervision_rot, rot_bt=rot_bt, corruption=corruption)
        
        self.gallery = MyData(root, label_txt=test_txt, transform=transform_Dict['center-crop'], mode=mode, self_supervision_rot=self_supervision_rot, rot_bt=rot_bt)




