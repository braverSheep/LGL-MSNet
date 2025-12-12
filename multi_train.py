import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.nn import functional as F
import os, random
from tqdm import *

import math
from My_model import *
from utils import *

from My_model import *
#from model import *
# from DGCNN import *

# 设置随机数种子
def seed_torch(seed=12):
    random.seed(seed)# 设置 Python 内置 random 模块的种子
    np.random.seed(seed)# 设置 NumPy 的随机种子
    torch.manual_seed(seed)# 设置 PyTorch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)# 设置 PyTorch 的 GPU 随机种子
    torch.cuda.manual_seed_all(seed)# 设置所有 GPU 的随机种子
    torch.backends.cudnn.benchmark = False  # 关闭自动选择最快算法的选项
    torch.backends.cudnn.deterministic = True # 让 CuDNN 以确定性的方式执行
seed_torch()
from data_input import getloader
# from data_input_rotate import getloader

# 参数正则化L2 防止模型过拟合
def parameter_Regular(New_model,lambada=0.0005):
    reg_loss = 0.0
    for param in New_model.parameters():
        reg_loss += torch.norm(param, p=2).to(device)
    return reg_loss * lambada

# 整个训练脚本产生的结果都将存入这个脚本中，模型，log等等
def cre_prolog():
    # 获取当前脚本的绝对路径  
    current_script_path = os.path.abspath(__file__)  
    # 获取当前脚本的所在目录  
    parent_dir = os.path.dirname(current_script_path)  
    # 定义要创建的Pro_log目录的路径
    pro_log_dir_path = os.path.join(parent_dir, 'Pro_log')  
      
    # 检查Pro_log目录是否存在，如果不存在则创建  
    if not os.path.exists(pro_log_dir_path):  
        os.makedirs(pro_log_dir_path)  
        print(f"Directory '{pro_log_dir_path}' created.")  

    return pro_log_dir_path


data_dir = "D:/shenduxuexi/SEED/seed_4s"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
exp_dir = cre_prolog()
log_file = f"{exp_dir}/log.txt"


def before_train_subject(num_subj=1,session_num=1,batch_size = 32,lr = 0.001,epochs = 240, which_local=0, tra_loader=None, tes_loader=None):
    best_acc = 0
    best_epoch = 0
    data_loader = getloader(data_dir, subj_num=num_subj, session_num=session_num, batch_size=batch_size, is_train=False, is_shuffle=False)
    labelTracker = LabelStabilityTracker(stability_threshold=0.05, stable_count_threshold=2)
    
    local_name = {0:'tran_fea1', 1:'tran_fea2', 2:'tran_fea3', 3:'tran_fea4'}

    net = local_stream(feature_dim=5, num_class=3, dim=[10, 20, 40, 80], num_heads=16, which_local=which_local).to(device)
    
    criterion_recon = nn.MSELoss()
    criterion_label = nn.CrossEntropyLoss()#交叉熵损失
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)# 0.0001 bs=128

    
    test_acc_best = 0
    max_test_acc = 0
    for epoch in range(epochs):
        tra_loss = 0
        tes_loss = 0
        tra_acc = 0
        tes_acc = 0
        net.train()
        for step, (images, labels) in enumerate(tra_loader):
            inputs = images[local_name[which_local]].to(device) 
            cla,x_recon, _ = net(inputs)

            loss = criterion_label(cla, labels.to(device)) + criterion_recon(x_recon, images[local_name[which_local]].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 训练集中的损失值和准确
            predict_y = torch.max(cla, dim=1)[1]
            tra_acc += (predict_y == labels.to(device)).sum().item()
            tra_loss += loss.item()
        
        tra_loss = tra_loss / len(tra_loader)
        tra_acc = tra_acc / len(tra_loader.dataset)

        label_trackr = None
        net.eval()
        for step, (images, labels) in enumerate(data_loader):
            inputs = images[local_name[which_local]].to(device) 
            cla, _ , _ = net(inputs)
                                 
            predict_y = torch.max(cla, dim=1)[1]
            tes_acc += (predict_y == labels.to(device)).sum().item()

            if label_trackr is None:
                label_trackr = predict_y
            else:
                label_trackr = torch.cat((label_trackr, predict_y), dim=0)
    

        tes_acc = tes_acc / len(data_loader.dataset)
        change_rate = labelTracker.compute_label_change_rate(label_trackr) 

        if test_acc_best < tes_acc:
                test_acc_best = tes_acc


        if labelTracker.is_stable():
            print(f'\n pretrain Epoch {epoch} tra_loss: {tra_loss:.4f}, tra_acc: {tra_acc:.4f}, tes_acc: {tes_acc:.4f}, test_acc_best: {test_acc_best:.4f}, ' , f"变化率 = {change_rate:.6f}, 稳定轮数 = {labelTracker.stable_count}")
            model_path_1 = f"{exp_dir}/pre_model_subj{num_subj}_local{which_local+1}.pth"
            torch.save(net.state_dict(), model_path_1)
            # break

    # model_path_1 = f"{exp_dir}/pre_model_subj{num_subj}_local{which_local+1}.pth"
    # torch.save(net.state_dict(), model_path_1)
                 
    return None
 

def train_subject(num_subj=1,session_num=1,batch_size = 32,lr = 0.001,epochs = 240,T_max = 40, drop_rate = 0,):
    tra_loader = getloader(data_dir, subj_num=num_subj, session_num=session_num, batch_size=batch_size, is_train=True, is_shuffle=True, num_workers=0)
    tes_loader = getloader(data_dir, subj_num=num_subj, session_num=session_num, batch_size=batch_size, is_train=False, is_shuffle=True, num_workers=0)
    # 定义网络、损失函数、优化器、迭代步骤
    # data_loader = getloader(data_dir, subj_num=num_subj, session_num=session_num, batch_size=batch_size, is_train=False, is_shuffle=False)

    before_train_subject(num_subj=num_subj, session_num=session_num, batch_size = 32, lr = 0.0008, epochs = epochs, which_local=0, tra_loader=tra_loader, tes_loader=tes_loader)
    before_train_subject(num_subj=num_subj, session_num=session_num, batch_size = 32, lr = 0.0008, epochs = epochs, which_local=1, tra_loader=tra_loader, tes_loader=tes_loader)
    before_train_subject(num_subj=num_subj, session_num=session_num, batch_size = 32, lr = 0.0008, epochs = epochs, which_local=2, tra_loader=tra_loader, tes_loader=tes_loader)
    before_train_subject(num_subj=num_subj, session_num=session_num, batch_size = 32, lr = 0.0008, epochs = epochs, which_local=3, tra_loader=tra_loader, tes_loader=tes_loader)

    net = global_stream().to(device)

    model_path_1 = f"{exp_dir}/pre_model_subj{num_subj}_local1.pth"
    net.local_1.load_state_dict(torch.load(model_path_1),strict=True)
    model_path_2 = f"{exp_dir}/pre_model_subj{num_subj}_local2.pth"
    net.local_2.load_state_dict(torch.load(model_path_2),strict=True)
    model_path_3 = f"{exp_dir}/pre_model_subj{num_subj}_local3.pth"
    net.local_3.load_state_dict(torch.load(model_path_3),strict=True)
    model_path_4 = f"{exp_dir}/pre_model_subj{num_subj}_local4.pth"
    net.local_4.load_state_dict(torch.load(model_path_4),strict=True)

    
    loss_function = nn.CrossEntropyLoss()
    criterion_recon = nn.MSELoss() # 均方误差

    optimizer = optim.Adam(net.parameters(), lr=lr)# 0.0001 bs=128, weight_decay=0.0001
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = T_max, eta_min=0, last_epoch=-1) # T_max=20：最大更新步数

    best_acc = 0
    best_epoch = 0

    for epoch in range(epochs):

        tra_loss = 0
        tes_loss = 0
        tra_acc = 0
        tes_acc = 0
        cla_1_acc = 0
        cla_2_acc = 0
        cla_3_acc = 0
        cla_4_acc = 0
        net.train()
        for step, data in enumerate(tra_loader):
            images, labels = data 
            cla_g,cla_1,cla_2,cla_3,cla_4, xg, xl = net(images)            

            optimizer.zero_grad()

            loss = loss_function(cla_g, labels.to(device))*10 + loss_function(cla_1, labels.to(device)) + loss_function(cla_2, labels.to(device)) + loss_function(cla_3, labels.to(device)) + loss_function(cla_4, labels.to(device)) + criterion_recon(xg, images['tran_fea_all'].cuda()) + criterion_recon(xl, images['tran_fea_all'].cuda())
            loss.backward()
            optimizer.step()            
                # 训练集中的损失值和准确
            predict_y = torch.max(cla_g, dim=1)[1].to(device)
            tra_acc += (predict_y == labels.to(device)).sum().item()
            tra_loss += loss.item()
            #局部流1准确率
            predict_y_1 = torch.max(cla_1, dim=1)[1].to(device)
            cla_1_acc += (predict_y_1 == labels.to(device)).sum().item()
            #局部流2准确率    
            predict_y_2 = torch.max(cla_2, dim=1)[1].to(device)
            cla_2_acc += (predict_y_2 == labels.to(device)).sum().item()
            #局部流3准确率
            predict_y_3 = torch.max(cla_3, dim=1)[1].to(device)
            cla_3_acc += (predict_y_3 == labels.to(device)).sum().item()
            #局部流4准确率
            predict_y_4 = torch.max(cla_4, dim=1)[1].to(device)
            cla_4_acc += (predict_y_4 == labels.to(device)).sum().item()
        # scheduler.step()
        tra_loss = tra_loss / len(tra_loader)
        tra_acc = tra_acc / len(tra_loader.dataset)
        cla_1_acc = cla_1_acc / len(tra_loader.dataset)
        cla_2_acc = cla_2_acc / len(tra_loader.dataset)
        cla_3_acc = cla_3_acc / len(tra_loader.dataset)
        cla_4_acc = cla_4_acc / len(tra_loader.dataset)

        net.eval()  # 在测试过程中关掉dropout方法，不希望在测试过程中使用dropout
        with torch.no_grad():
            for step, data in enumerate(tes_loader):
                images, labels = data
                cla_g,cla_1,cla_2,cla_3,cla_4, xg, xl = net(images)
               
                predict_y = torch.max(cla_g, dim=1)[1]
                tes_acc += (predict_y == labels.to(device)).sum().item()
                tes_loss += loss.item()

            tes_loss = 0 / len(tes_loader)
            tes_acc = tes_acc / len(tes_loader.dataset)
            if tes_acc > best_acc:
                best_acc = tes_acc
                best_epoch = epoch + 1 
                torch.save(net.state_dict(), f"{exp_dir}/best_model_subj{num_subj}_session{session_num}.pth")#_acc{tes_acc:.4f}
            
            # 在进度条上更新当前 epoch 及最好的验证精度

        print(f'\n Epoch {epoch} tra_loss: {tra_loss:.4f}, tra_acc: {tra_acc:.4f}, tes_acc: {tes_acc:.4f}, tes_loss: {tes_loss:.4f}, best_acc: {best_acc:.4f}, best_epoch: {best_epoch} ,cla_1_acc: {cla_1_acc:.4f} ,cla_2_acc: {cla_2_acc:.4f} ,cla_3_acc: {cla_3_acc:.4f} ,cla_4_acc: {cla_4_acc:.4f}')
        with open(log_file, 'a') as file:  
            line_to_write = f'Epoch {epoch} tra_loss: {tra_loss:.4f}, tra_acc: {tra_acc:.4f}, tes_acc: {tes_acc:.4f}, tes_loss: {tes_loss:.4f}, best_acc: {best_acc:.4f}, best_epoch: {best_epoch} ,cla_1_acc: {cla_1_acc:.4f} ,cla_2_acc: {cla_2_acc:.4f} ,cla_3_acc: {cla_3_acc:.4f} ,cla_4_acc: {cla_4_acc:.4f}\n'  
            file.write(line_to_write) 
        if tes_acc>0.9999:
            break

    return best_acc, best_epoch

if __name__ == '__main__':

    # 超参数设置
    batch_size = 32             #每次训练输入模型的数据量
    lr = 0.0083 
    epoch = 180                 #训练总轮数
    T_max = 18                #控制学习率在一个完整周期内的变化范围和速度
    drop_rate = 0              #不使用Dropout
  
    # 开始训练
    best_Acc = []
    for i in range(1, 16):
        print(f'------------------start subect:{i}--------------------- ')
        with open(log_file, 'a') as file:  
            file.write(f'\n---------------------------------------------------------------------start subect:{i}---------------------------------------------------------------------\n ')
        best_model_acc, best_epoch = train_subject(num_subj=i, session_num=1, batch_size=batch_size, lr=lr, epochs=epoch, T_max=T_max, drop_rate=drop_rate,)
        best_Acc.append(best_model_acc)

    k = 0
    result_string = ''
    for i in best_Acc:
        k=k+1                         
        str1 = f'subject{k} acc:{i} \n'
        result_string = ''.join([result_string, str1])

    # 所有subjects的平均准确率、标准差
    mean_acc = np.mean(best_Acc)
    sta = np.std(best_Acc, ddof=1)
    mean_str = f'mean_acc:{mean_acc} sta:{sta}'

    result_string = ''.join([result_string, mean_str]) 

    with open(log_file, 'a') as file:  
            file.write(result_string)
    print(result_string)