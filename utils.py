import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch

class LabelStabilityTracker:
    def __init__(self, stability_threshold=0.01, stable_count_threshold=3):
        """
        初始化稳定性追踪器

        参数:
        stability_threshold (float): 标签变化率阈值（0~1），低于该值视为稳定
        stable_count_threshold (int): 连续稳定轮数达到该值视为真正稳定
        """
        self.stability_threshold = stability_threshold
        self.stable_count_threshold = stable_count_threshold
        self.prev_labels = None
        self.stable_count = 0

    def compute_label_change_rate(self, current_labels: torch.Tensor):
        """
        比较当前标签与上一轮标签的变化率，并更新状态

        参数:
        current_labels (torch.Tensor): 当前预测标签（1D张量）

        返回:
        float: 标签变化率（0~1之间）
        """
        if self.prev_labels is None:
            print("[Tracker] 第一次调用，初始化标签")
            change_rate = 1.0  # 初始认为全变
        else:
            if current_labels.shape != self.prev_labels.shape:
                raise ValueError(f"[Tracker] 标签长度不一致: 当前={current_labels.shape}, 之前={self.prev_labels.shape}")
            num_changes = torch.sum(self.prev_labels != current_labels).item()
            change_rate = num_changes / len(current_labels)

        # 更新状态
        self.prev_labels = current_labels.clone().detach()

        if change_rate < self.stability_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0

        return change_rate

    def is_stable(self):
        """
        判断是否达到稳定条件

        返回:
        bool: True 表示已稳定
        """
        return self.stable_count >= self.stable_count_threshold




# 统计预测标签的预测稳定性
# class LabelStabilityTracker: # 这个是有问题的，因为数据加载的时候是随机采样的，因此就算每次都预测成功，但标签的顺序也是不一样的
#     def __init__(self, stability_threshold=0.05, stable_count_threshold=3):
#         """
#         初始化稳定性追踪器
        
#         参数:
#         stability_threshold (float): 标签变化率阈值，低于该值视为稳定（如 0.05 表示 5%）
#         stable_count_threshold (int): 连续稳定的次数阈值，用于判定稳定（如 3 表示连续 3 轮稳定）
#         """
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.prev_labels = None
#         self.stability_threshold = stability_threshold# 标签变化率阈值
#         self.stable_count_threshold = stable_count_threshold# 连续达到稳定的轮数阈值
#         self.stable_count = 0# 当前连续稳定的轮数
    
#     def compute_label_change_rate(self, current_labels):
#         """
#         计算当前标签相对于上一轮的变化率
        
#         参数:
#         current_labels (torch.Tensor): 当前轮的预测标签，形状为 [num_samples]
        
#         返回:
#         float: 标签变化率，表示前后两轮标签不一致的比例
#         """
#         if self.prev_labels is None:
#             # 第一轮时没有上一轮标签，变化率设为 0
#             change_rate = 0.0
#         else:
#             num_changes = torch.sum(self.prev_labels != current_labels).item()
#             change_rate = num_changes / len(current_labels) 
#         #比较上次标签和当前标签，统计不一致的数量并归一化            
#         # 更新上一轮标签
#         self.prev_labels = current_labels.clone()#.clone() 保证不会受外部变量更改影响
        
#         # 判断是否达到稳定性
#         if change_rate < self.stability_threshold:
#             self.stable_count += 1
#         else:
#             self.stable_count = 0
#         # print("prev_labels[:10]:", self.prev_labels[:10].tolist() if self.prev_labels is not None else "None")
#         # print("curr_labels[:10]:", current_labels[:10].tolist())

#         # 返回变化率
#         return change_rate
    
#     def is_stable(self):
#         """
#         检查是否达到连续稳定的阈值，用于停止条件
        
#         返回:
#         bool: 若达到连续稳定的次数阈值，返回 True；否则返回 False
#         """
#         print(self.stable_count, self.stable_count_threshold)
#         return self.stable_count >= self.stable_count_threshold

#     def track_stability(self, model, test_loader):
#         """
#         计算并返回标签变化率
        
#         参数:
#         model (torch.nn.Module): 训练中的模型
#         test_loader (DataLoader): 测试集的数据加载器
        
#         返回:
#         float: 当前轮标签变化率
#         """
#         model.eval()  # 设置模型为评估模式
#         current_labels = []
        
#         acc = 0
#         tes_acc = 0
#         with torch.no_grad():
#             for data, labels in test_loader:
#                 inputs = data.to(self.device)  # 确保数据和模型在同一设备
#                 outputs, _ = model(inputs)
#                 preds = outputs.argmax(dim=1)  # 获取预测标签
#                 current_labels.append(preds)

#                 # 预测准确率
#                 tes_acc += (preds == labels.to(self.device)).sum().item()
#         tes_acc = tes_acc / len(test_loader.dataset)

#         current_labels = torch.cat(current_labels)#将所有批次的预测标签拼接成一个完整的张量
#         change_rate = self.compute_label_change_rate(current_labels)#计算标签变化率
        
#         # 回到训练模式
#         model.train()
        
#         return change_rate, tes_acc


