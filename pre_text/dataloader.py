import torch
from torch.utils.data import Dataset, DataLoader

import json
import clip


def datafold_read(datalist):
    with open(datalist) as f:
        json_data = json.load(f)

    return json_data


class BraTsDataset(Dataset):
    def __init__(self, data_list, phase='train'):
        super(BraTsDataset, self).__init__()
        self.data_list = data_list[phase]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        token = []
        buffer = []
    
        for i, j, k in zip(self.data_list[item]['wt'], self.data_list[item]['tc'], self.data_list[item]['et']):
            if i is None:
                i = [0] * 5
                
            if j is None:
                j = [0] * 5
                
            if k is None:
                k = [0] * 5
                
            buffer = i + j + k
            buffer = clip.tokenize(str(buffer)).to(torch.float32).squeeze(0)
            token.append(buffer)
    
        return {
            'token_0': token[0],
            'token_1': token[1],
            'token_2': token[2]
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
                                shuffle=False)
    else: 
        dataloader = DataLoader(datasets,
                            batch_size=batch_size,
                            num_workers=num_works,
                            pin_memory=True,
                            shuffle=True)

    return dataloader
    
if __name__ == "__main__":
    train_file_path = '/home/qlc/raid/dataset/Brats2023/ag_2d/clip_consis_train.json'
    # val_file_path = '/home/qlc/raid/dataset/Brats2023/ag_2d/ag_t1_val.json'
    # test_file_path = '/home/qlc/raid/dataset/Brats2023/ag_2d/ag_t1_test.json'
    
    dataloader = get_loader(train_file_path, 4, 4, phase='train')
    
    for i, data in enumerate(dataloader):
        token_0 = data['token_0']
        token_1 = data['token_1']
        token_2 = data['token_2']
        print(token_0.size(), token_1.size(), token_2.size())
        assert 1 == 2
    


