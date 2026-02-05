import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from monai import transforms
import json
import os
import clip


def datafold_read(datalist):
    with open(datalist) as f:
        json_data = json.load(f)

    return json_data


class BraTsDataset(Dataset):
    def __init__(self, data_list, phase='train', modality='t1c'):
        super(BraTsDataset, self).__init__()
        self.data_list = data_list[phase]

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensord(keys=["gli", "image"]),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.ToTensord(keys=["gli", "image"]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensord(keys=["gli", "image"]),
            ]
        )

        self.phase = phase
        self.modality = modality

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        data = self.load_image(self.data_list[item])
        if self.phase == 'train':
            data = self.train_transform(data)
        elif self.phase == 'val':
            data = self.val_transform(data)
        elif self.phase == 'test':
            data = self.test_transform(data)

        return data

    def load_image(self, file_dic):
        data = np.load(file_dic['path'])
        
        image = data[self.modality]
        label = data['label']
        
        if file_dic['wt'] is None:
            file_dic['wt'] = [0] * 5

        if file_dic['tc'] is None:
            file_dic['tc'] = [0] * 5
            
        if file_dic['et'] is None:
            file_dic['et'] = [0] * 5
        
        token = file_dic['wt'] + file_dic['tc'] + file_dic['et']
        token = clip.tokenize(str(token)).to(torch.float32).squeeze(0)

        return {
            'image': image,
            'gli': label,
            'token': token,
            'path': os.path.split(file_dic['path'])[-1]
        }
        

def get_loader(datalist_json,
               batch_size,
               num_works,
               modality='t1c',
               phase=None):

    files = datafold_read(datalist=datalist_json)
    
    datasets = BraTsDataset(data_list=files, phase=phase, modality=modality)

    if phase != 'train':
        dataloader = DataLoader(datasets,
                                batch_size=batch_size,
                                num_workers=num_works,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=True)
    else:
        dataloader = DataLoader(datasets,
                                batch_size=batch_size,
                                num_workers=num_works,
                                pin_memory=True,
                                shuffle=True,
                                drop_last=True)
    return dataloader
    
if __name__ == "__main__":
    # train_file_path = '/home/qlc/raid/dataset/Brats2023/ag_2d/ag_t1_train.json'
    # # val_file_path = '/home/qlc/raid/dataset/Brats2023/ag_2d/ag_t1_val.json'
    test_file_path = '/home/qlc/raid/dataset/Brats2023/ag/gli_gen.json'
    
    dataloader = get_loader(test_file_path, 4, 4, phase='test')
    
    for i, data in enumerate(dataloader):
        label = data['gli']
        image = data['image']
        print(label.shape, image.shape)   
    print(label.type())
    path = '/home/qlc/raid/dataset/Brats2023/ag/train/BraTS-GLI-00000-000/BraTS-GLI-00000-000-52.npz'
    data = np.load(path)

    label = data['label']
    image = data['t1c']
    print(label.shape, image.shape)    