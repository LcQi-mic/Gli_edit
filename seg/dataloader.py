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
    def __init__(self, data_list, phase='train'):
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
        
        t1c = data['t1c']
        t1n = data['t1n']
        t2w = data['t2w']
        t2f = data['t2f']
        
        image = np.stack([t1c, t1n, t2w, t2f], axis=0).reshape(4, 256, 256)
        
        label = data['label']
        
        return {
            'image': image,
            'gli': label,
            'path': os.path.split(file_dic['path'])[-1]
        }
        

def get_loader(datalist_json,
               batch_size,
               num_works,
               phase=None):

    files = datafold_read(datalist=datalist_json)
    
    datasets = BraTsDataset(data_list=files, phase=phase)

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
    train_file_path = '/home/qlc/raid/dataset/Brats2023/ag/mri_gen.json'
    # val_file_path = '/home/qlc/raid/dataset/Brats2023/ag_2d/ag_t1_val.json'
    # test_file_path = '/home/qlc/raid/dataset/Brats2023/ag_2d/ag_t1_test.json'
    
    dataloader = get_loader(train_file_path, 4, 4, modality='t1c', phase='train')
    
    for i, data in enumerate(dataloader):
        img = data['image']
        gli = data['gli']
        print(img.size(), gli.size())
        assert 1 == 2

    