import numpy as np
import torch
import os
import scipy
from torch.utils.data import Dataset, DataLoader


SEED_data_dir = "D:/shenduxuexi/SEED/seed_4s"

seesion1_label = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
seesion2_label = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
seesion3_label = [1,0,-1,-1,0,1,-1,0,1,1,0,-1,0,1,-1]
labels = [seesion1_label, seesion2_label, seesion3_label]

class  SEED_dataset(Dataset):
    def __init__(self, SEED_data_dir, subj_num, session_num, datasets='train'):
        super(SEED_dataset, self).__init__()
        self.datasets = datasets
        self.session_num = session_num
        # 候选测试样本
        self.train_Trails = []
        self.test_Trails = []
        self.val_Trails = []

        self.sample_list = []
        self.sample_label_list = []

        self.channel_group1 = [2 ,1 ,3, 4, 5, 7, 8, 9, 10, 11, 12, 13, ]
        self.channel_group2 = [6, 14, 15, 16, 22, 23, 24, 25, 31, 32, 33, 34, 40, 41, 42, 43, 49, 50 ]
        self.channel_group3 = [17, 18, 19, 20, 21, 26, 27, 28, 29, 30, 35, 36, 37, 38, 39, 44, 45, 46, 47, 48]
        self.channel_group4 = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
        
        self.channel_group_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                                    41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

        #获取指定受试者的路径
        session_dir = os.path.join(SEED_data_dir,str(self.session_num))
        for subj_file in os.listdir(session_dir):
            subj=int(subj_file.split("_")[0])
            if subj == subj_num:
                subj_file_name = subj_file
                break
        file_path = os.path.join(session_dir, subj_file_name)

        for i in range(0,9):
            self.train_Trails.append(i)
        
        for i in range(9,15):
            self.test_Trails.append(i)

        # print(self.train_Trails, self.test_Trails)

        if self.datasets == 'train':
            for i in range(len(self.train_Trails)):
                trial = self.train_Trails[i]
                # print(trial)
                X_de = scipy.io.loadmat(file_path)['de_LDS{}'.format(trial + 1)]
                # 首先进入对应session,然后再session里面获得对应标签
                y = int(labels[self.session_num-1][trial]) + 1
                
                # 从第二个维度进行划分，42个
                for t in range(X_de.shape[1]):
                    x_de = torch.tensor(X_de[:, t, :]).float()
                    # 单个样本数据预处理
                    x_de = self.eeg_transformer(x_de, "Standardization")
                    
                    # 构建字典样本，包含不同通道组的特征
                    sample = {
                        'tran_fea1': self.gettran(x_de, self.channel_group1),
                        'tran_fea2': self.gettran(x_de, self.channel_group2),
                        'tran_fea3': self.gettran(x_de, self.channel_group3),
                        'tran_fea4': self.gettran(x_de, self.channel_group4),
                        'tran_fea_all': self.gettran(x_de, self.channel_group_all),
                    }

                    # 将样本和标签加入列表
                    self.sample_list.append(sample)
                    self.sample_label_list.append(y)

        elif self.datasets == 'test':
            for i in range(len(self.test_Trails)):
                trial = self.test_Trails[i]
                # print(trial)
                X_de = scipy.io.loadmat(file_path)['de_LDS{}'.format(trial + 1)]
                y = int(labels[self.session_num-1][trial]) + 1
                for t in range(X_de.shape[1]):
                    x_de = torch.tensor(X_de[:, t, :]).float()
                    x_de = self.eeg_transformer(x_de,"Standardization")
                
                    # 构建字典样本，包含不同通道组的特征
                    sample = {
                        'tran_fea1': self.gettran(x_de, self.channel_group1),
                        'tran_fea2': self.gettran(x_de, self.channel_group2),
                        'tran_fea3': self.gettran(x_de, self.channel_group3),
                        'tran_fea4': self.gettran(x_de, self.channel_group4),
                        'tran_fea_all': self.gettran(x_de, self.channel_group_all),
                    }

                    # 将样本和标签加入列表
                    self.sample_list.append(sample)
                    self.sample_label_list.append(y)

    
    def eeg_transformer(self, x, data_process):
        if data_process == "Standardization" :
            return (x - x.mean()) / x.std()
        # if data_process == "min_max_normalization":
        #     return (x - x.min()) / (x.max - x.min() + 1e-8)
        
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        return self.sample_list[index], self.sample_label_list[index]

    def gettran(self, x, channel_selected):
        # print('channel_selected', len(channel_selected))
        channel_selected = [x - 1 for x in channel_selected]
        # print(len(channel_selected))
        return x[channel_selected, :]


# def getloader(SEED_data_dir, subj_num=1, session_num=1, batch_size=8, datasets='train', is_drop_last=False, num_workers=0):
#     gen_dataset = SEED_dataset(SEED_data_dir, subj_num, session_num, datasets)
#     # print(gen_dataset.__len__())
#     return DataLoader(
#             gen_dataset,
#             batch_size=batch_size,
#             shuffle = True if datasets=='train' else False, 
#             num_workers=num_workers, 
#             drop_last=is_drop_last
#         )

def getloader(SEED_data_dir, subj_num=1, session_num=1, batch_size=8, is_train=False, is_drop_last=False, is_shuffle=False, num_workers=0):
    gen_dataset = SEED_dataset(SEED_data_dir, subj_num, session_num, datasets='train' if is_train else 'test')
    # print(True if is_train else False)
    # print(gen_dataset.__len__())
    return DataLoader(
            gen_dataset,
            batch_size=batch_size,
            shuffle = is_shuffle, 
            num_workers=0, 
            drop_last=is_drop_last,
            # worker_init_fn=worker_init_fn, 
            # generator=g
        )
    
if __name__== "__main__":

    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_loader = getloader(SEED_data_dir, subj_num=1, session_num=1, batch_size=32, is_train=False, is_drop_last=False, is_shuffle=False, num_workers=0)
    # NL_data, LL_data, labels = data['NL'].to(device),  data['LL'].to(device), data["Label"].to(device)

    for step, data in enumerate(data_loader, start=0):
        data, labels = data
        print(step, data['tran_fea1'].size(), data['tran_fea2'].size(), data['tran_fea3'].size(), 
              data['tran_fea4'].size(), data['tran_fea_all'].size(), labels.size())
        break
