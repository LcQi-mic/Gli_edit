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
        # self.data_list = data_list[phase]
        self.data_list = data_list
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
    test_file_path = '/Users/qlc/Desktop/Dataset/ag/mri_gen_test.json'
    
    dataloader = get_loader(test_file_path, 1, 4, phase='train')
    
    wt_size = []
    et_size = []
    tc_size = []
    
    for i, data in enumerate(dataloader):
        # img = data['image']
        gli = data['gli']

        wt = int(np.sum(gli[0][0].numpy()))
        tc = int(np.sum(gli[0][1].numpy()))
        et = int(np.sum(gli[0][2].numpy()))
        print(wt, tc, et)
        wt_size.append(wt)
        tc_size.append(tc)
        et_size.append(et)

    np.save('wt.npy', wt_size)
    np.save('tc.npy', tc_size)
    np.save('et.npy', et_size)


    def remove_outliers_numpy(data, percentile=5):
        """
        删除前percentile%和后percentile%的数据
        """
        data_array = np.array(data)
        
        # 计算分位数
        lower_bound = np.percentile(data_array, percentile)
        upper_bound = np.percentile(data_array, 100 - percentile)
        
        # 筛选在范围内的数据
        filtered_data = data_array[(data_array >= lower_bound) & (data_array <= upper_bound)]
        
        return filtered_data.tolist()

    # wt = np.load('wt.npy')
    # wt = remove_outliers_numpy(wt)
    # print(wt)

    # tc = np.load('tc.npy')
    # tc = remove_outliers_numpy(tc)
    # print(tc)
    
    et = np.load('et.npy')
    et = remove_outliers_numpy(et)
    print(et)