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
                transforms.ToTensord(keys=["gli_0", "gli_1"]),
            ]
        )
        self.val_transform = transforms.Compose(
            [
                transforms.ToTensord(keys=["gli_0", "gli_1"]),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.ToTensord(keys=["gli_0", "gli_1"]),
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
        data = np.load(file_dic['path'][0])
    
        label_0 = data['label']

        if file_dic['wt'][0] is None:
            file_dic['wt'][0] = [0] * 5

        if file_dic['tc'][0] is None:
            file_dic['tc'][0] = [0] * 5
            
        if file_dic['et'][0] is None:
            file_dic['et'][0] = [0] * 5
        
        token_0 = file_dic['wt'][0] + file_dic['tc'][0] + file_dic['et'][0]
        token_0 = clip.tokenize(str(token_0)).to(torch.float32).squeeze(0)
        
        data = np.load(file_dic['path'][1])
        
        label_1 = data['label']
        
        if file_dic['wt'][1] is None:
            file_dic['wt'][1] = [0] * 5

        if file_dic['tc'][1] is None:
            file_dic['tc'][1] = [0] * 5
            
        if file_dic['et'][1] is None:
            file_dic['et'][1] = [0] * 5
        
        token_1 = file_dic['wt'][1] + file_dic['tc'][1] + file_dic['et'][1]
        token_1 = clip.tokenize(str(token_1)).to(torch.float32).squeeze(0)

        return {
            'gli_0': label_0,
            'gli_1': label_1,
            'token_0': token_0,
            'token_1': token_1,
            'path': os.path.split(file_dic['path'][0])[-1]
        }
        

def get_loader(datalist_json,
               batch_size,
               num_works,
               modality,
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
    train_file_path = '/home/qlc/raid/dataset/Brats2023/ag_2d/ag_t1_train.json'
    # val_file_path = '/home/qlc/raid/dataset/Brats2023/ag_2d/ag_t1_val.json'
    # test_file_path = '/home/qlc/raid/dataset/Brats2023/ag_2d/ag_t1_test.json'
    
    dataloader = get_loader(train_file_path, 4, 4, phase='train')
    
    for i, data in enumerate(dataloader):
        img = data['image']
        gli = data['gli']
        print(img.size(), gli.size())
        assert 1 == 2