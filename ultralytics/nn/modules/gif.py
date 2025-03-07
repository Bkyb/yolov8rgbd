import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ('Gif')

class Gif(nn.Module):
    def __init__(self, channels):
        super(Gif, self).__init__()
        # 두 특징 맵을 concat하니까 채널 크기가 2배				
        # FG를 계산할 때 사용되는 3x3x1 컨볼루션 레이어
            
    def forward(self, x):
        # 먼저 FG를 계산하자
        # 두 모달리티의 특징 맵을 concat한다. x = [F1, F2]
        F1, F2 = x
        FG = torch.cat(x, dim=1)  

        if self.conv_w1 is None:
            channels = FG.size(1)
            self.conv_w1 = nn.Conv2d(channels, 1, kernel_size=3, padding=1).to(FG.device)
            self.conv_w2 = nn.Conv2d(channels, 1, kernel_size=3, padding=1).to(FG.device)
            self.conv_fusion = nn.Conv2d(channels, channels, kernel_size=1).to(FG.device)

        
        # Weight Generation Network:
        w1 = torch.sigmoid(self.conv_w1(FG))
        w2 = torch.sigmoid(self.conv_w2(FG))
        
        # Element-Wise Product:
        F1_weighted = torch.mul(F1, w1)
        F2_weighted = torch.mul(F2, w2)
        
        # Concat both weighted feature map
        FF = torch.cat([F1_weighted, F2_weighted], dim=1)
        
        # 마지막에 ReLU를 쓸지 안쓸지는 해봐야 알 듯? 원래는 ReLU가 들어감
        FJ = F.relu(self.conv_fusion(FF))
        # FJ = self.conv_fusion(FF)
        
        return FJ
