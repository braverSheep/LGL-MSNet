import torch
import torch.nn as nn
import torch.nn.functional as F

from positional_encodings.torch_encodings import PositionalEncoding1D#位置嵌入

channel_group1 = [2 ,1 ,3, 4, 5, 7, 8, 9, 10, 11, 12, 13, ]
channel_group2 = [6, 14, 15, 16, 22, 23, 24, 25, 31, 32, 33, 34, 40, 41, 42, 43, 49, 50]
channel_group3 = [17, 18, 19, 20, 21, 26, 27, 28, 29, 30, 35, 36, 37, 38, 39, 44, 45, 46, 47, 48]
channel_group4 = [51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
channel_groups = [channel_group1, channel_group2, channel_group3, channel_group4]


class autocoder(nn.Module):
    def __init__(self, channels, out_channels):
        super(autocoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            )
       

        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=channels, kernel_size=1, padding=0),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):

        f = self.encoder(x)
        x = self.decoder(f)

        return f, x 


class TADNet(nn.Module):
    def __init__(self, feature_dim=5, classs_num=4, channel_n=62):
        super(TADNet, self).__init__()
        out_channels = channel_n #// 2

        self.autocoder = autocoder(channel_n,out_channels)

        self.classfier = nn.Sequential(
            nn.Linear(channel_n*feature_dim, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(64),
            nn.Dropout(p=0.5),
            nn.Linear(64, classs_num)
        )

    def forward(self, x):
        # x = data['tran_fea_all'].cuda()
        b, c, d = x.shape[0], x.shape[1], x.shape[2]  
        f, x = self.autocoder(x)

        cla = f.view(b,-1)
        cla = self.classfier(cla) 

        return cla, x,f



class local_stream(nn.Module):
    def __init__(self, feature_dim=5, num_class=3, dim = [10, 20, 40, 80], num_heads = 16, which_local = 0):

        super(local_stream, self).__init__()
        # 参数设置
        self.channel_n = [12, 18, 20, 12, 62]
        self.dim = dim                          #输入特征的维度

        self.trans1 = TADNet(feature_dim=5, classs_num=num_class, channel_n=self.channel_n[which_local])

        
    def forward(self, x):
        cla, x, f = self.trans1(x)
        return cla, x, f            


class LGL_Block(nn.Module):
    def __init__(self, dim):
        super(LGL_Block, self).__init__()
        self.dim = dim
        # 前四个区域的通道数 + 全局通道数
        self.channel_n = [12, 18, 20, 12, 62]
        self.sigmoid = nn.Sigmoid()
        # 四个区域各自的权重生成器
        self.uu = nn.ModuleList([ nn.Linear(self.dim * channel, 1) for channel in self.channel_n[:4] ])

    def _compute_confidence(self, logits):
        """计算置信度 = 最大概率 × (1 - 归一化熵)，返回 shape: [B, 1]"""
        probs = F.softmax(logits, dim=1)  # [B, num_classes]
        max_prob = probs.max(dim=1)[0]  # [B]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)  # [B]
        norm_entropy = entropy / torch.log(torch.tensor(probs.size(1), device=logits.device))  # [B]
        confidence = max_prob * (1 - norm_entropy)  # [B]
        return confidence.unsqueeze(1)  # [B, 1]

    def forward(self,
                x_1: torch.Tensor,
                x_2: torch.Tensor,
                x_3: torch.Tensor,
                x_4: torch.Tensor,
                g:   torch.Tensor,
                logits_1: torch.Tensor,
                logits_2: torch.Tensor,
                logits_3: torch.Tensor,
                logits_4: torch.Tensor,
                channel_groups: list):
        B = x_1.shape[0]

        # Step 1: 各局部流的复合置信度
        confs = torch.cat([
            self._compute_confidence(logits_1),
            self._compute_confidence(logits_2),
            self._compute_confidence(logits_3),
            self._compute_confidence(logits_4)
        ], dim=1)  # shape: [B, 4]

        # Step 2: 归一化置信度（行归一化），作为最终权重系数
        norm_conf = F.softmax(confs, dim=1)  # shape: [B, 4]

        # Step 3: 各区域融合权重（使用线性层 + sigmoid）
        u_base = [
            self.sigmoid(self.uu[0](x_1.view(B, -1))),
            self.sigmoid(self.uu[1](x_2.view(B, -1))),
            self.sigmoid(self.uu[2](x_3.view(B, -1))),
            self.sigmoid(self.uu[3](x_4.view(B, -1)))
        ]  # list of 4 tensors, shape: [B,1]

        # Step 4: 加权置信度反哺（最终融合因子）
        u_final = [
            u_base[0] * norm_conf[:, 0:1],
            u_base[1] * norm_conf[:, 1:2],
            u_base[2] * norm_conf[:, 2:3],
            u_base[3] * norm_conf[:, 3:4]
        ]  # 每个: [B,1]

        


        # 4. 根据 channel_groups 切出全局特征中的各区域
        x_region = []
        for group in channel_groups:
            indices = [i - 1 for i in group]
            x_region.append(g[:, indices, :])

        # 5. 加权融合局部 & 全局特征
        # print(x_1.shape, x_region[0].shape)
        x1_total = x_1 * u_final[0].unsqueeze(1) + x_region[0] * u_final[0].unsqueeze(1)
        x2_total = x_2 * u_final[1].unsqueeze(1) + x_region[1] * u_final[1].unsqueeze(1)
        x3_total = x_3 * u_final[2].unsqueeze(1) + x_region[2] * u_final[2].unsqueeze(1)
        x4_total = x_4 * u_final[3].unsqueeze(1) + x_region[3] * u_final[3].unsqueeze(1)

        #6. 重新组合成 (B,62,dim)，使用 index_copy 保留梯度
        device = g.device
        trans = torch.zeros_like(g)  # shape = (B,62,dim), same device & dtype
        for x_total, group in zip(
                [x1_total, x2_total, x3_total, x4_total],
                channel_groups[:4]):
            idx = torch.tensor([i - 1 for i in group],
                               dtype=torch.long,
                               device=device)
            # 在 dim=1 上，把 x_total 放到对应的通道位置
            trans = trans.index_copy(dim=1, index=idx, source=x_total)

        return norm_conf[:,0:1], norm_conf[:,1:2], norm_conf[:,2:3], norm_conf[:,3:4], trans



class global_stream(nn.Module):
    def __init__(self, feature_dim=5, num_class=3, dim = [10, 20, 20, 10], num_heads = 32):

        super(global_stream, self).__init__()

        self.dim = [10, 20, 20, 10] 
        self.channel_n = [12, 18, 20, 12, 62]#每个通道组的通道数量，包括 4 个区域通道组和一个全局通道组


        self.local_1 = local_stream(feature_dim=feature_dim, num_class=num_class, dim = self.dim, num_heads = num_heads,which_local = 0)
        self.local_2 = local_stream(feature_dim=feature_dim, num_class=num_class, dim = self.dim, num_heads = num_heads,which_local = 1)
        self.local_3 = local_stream(feature_dim=feature_dim, num_class=num_class, dim = self.dim, num_heads = num_heads,which_local = 2)
        self.local_4 = local_stream(feature_dim=feature_dim, num_class=num_class, dim = self.dim, num_heads = num_heads,which_local = 3)

        # out_channels = channel_n #// 2
        self.autocoder = autocoder(62,62)

        self.classfier = nn.Sequential(
            nn.Linear(feature_dim*62, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            # nn.BatchNorm1d(64),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_class)
        )


        self.L1 = LGL_Block(feature_dim)




    def forward(self,data):
        
        cla_1, x1, f1 = self.local_1(data['tran_fea1'].cuda()) # cla_1局部流的分类结果，x1局部流的重构结果，f1局部流中间层输出结果
        cla_2, x2, f2 = self.local_2(data['tran_fea2'].cuda())
        cla_3, x3, f3 = self.local_3(data['tran_fea3'].cuda())
        cla_4, x4, f4 = self.local_4(data['tran_fea4'].cuda())

        fg, xg = self.autocoder(data['tran_fea_all'].cuda())

        u1,u2,u3,u4,fg = self.L1(f1,f2,f3,f4,fg, cla_1, cla_2, cla_3, cla_4,channel_groups)

        # 全局流分类
        fg = fg.view(fg.shape[0], -1)
        cla_g = self.classfier(fg)

        # 局部流的重构融合 
        x_region = []
        for group in channel_groups:
            indices = [i - 1 for i in group]
            x_region.append(xg[:, indices, :])


        # 重新组合成 (B,62,dim)，使用 index_copy 保留梯度
        device = xg.device
        x_local = torch.zeros_like(xg)  # shape = (B,62,dim), same device & dtype
        for x_total, group in zip(
                [x1, x2, x3, x4],
                channel_groups[:4]):
            idx = torch.tensor([i - 1 for i in group],dtype=torch.long,device=device)

            # 在 dim=1 上，把 x_total 放到对应的通道位置
            x_local = x_local.index_copy(dim=1, index=idx, source=x_total)

      
        return cla_g, cla_1, cla_2, cla_3, cla_4, xg, x_local

if __name__=='__main__':

    #net = New_model().to('cpu')
    net = global_stream().cuda()
    from data_input import getloader
    SEED_data_dir = "D:/shenduxuexi/SEED/seed_4s"

    data_loader = getloader(SEED_data_dir, subj_num=1, session_num=1, batch_size=32, is_train=True, is_drop_last=False, is_shuffle=True)
    for step,data in enumerate(data_loader,start=0):
        data,labels=data
        y1,y2,y3,y4,y5,y6,y7=net(data)
        #y1,_,_,_,_=net(data)#_表示忽略
        print(y1.shape, y2.shape, y6.shape, y7.shape)
        break
