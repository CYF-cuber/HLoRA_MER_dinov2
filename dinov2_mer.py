import cv2
import argparse
import torch
import torch.nn as nn
import loralib as lora
import sys
sys.path.append("./dinov2_lora-main/")
from dinov2.models.vision_transformer import vit_giant2
sys.path.append("./GraphAttention/")
from models import GAT
import numpy as np
from torchsummary import summary

class dinov2_lora_GAT_MER_3cls(nn.Module):
    def __init__(self,lora_depth):
        super(dinov2_lora_GAT_MER_3cls, self).__init__()
        self.dinov2 = vit_giant2(patch_size=14, img_size=526, init_values=1.0, num_register_tokens=4, block_chunks=0,lora_depth = lora_depth )
        self.dinov2.load_state_dict(torch.load('/home/szw/.cache/torch/hub/checkpoints/dinov2_vitg14_pretrain.pth'), strict=False) #dinov2 giant version
        lora.mark_only_lora_as_trainable(self.dinov2)
        self.spfa_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4, 4), stride=(1, 1))
        self.spfa_pooling = nn.MaxPool2d(kernel_size=(5, 5), stride =(2,1))
        #self.weighting_conv_2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 8), stride=(1, 1))
        self.inner_pooling = nn.MaxPool2d(kernel_size=(5, 8), stride=(2,1))

        self.cls = nn.Linear(in_features=1531, out_features=3)
        self.iffa = GAT(1536,1000,1536,0.1,0.01,3)
        self.adj = torch.from_numpy(np.array(
                                            [[1,1,0,0,0,0,0,0],
                                             [1,1,1,0,0,0,0,0],
                                             [0,1,1,1,0,0,0,0],
                                             [0,0,1,1,1,0,0,0],
                                             [0,0,0,1,1,1,0,0],
                                             [0,0,0,0,1,1,1,0],
                                             [0,0,0,0,0,1,1,1],
                                             [0,0,0,0,0,0,1,1]])).cuda()
        
    
    def forward(self , x):
        B,C,T,H,W = x.shape
        frame_features = []
        for t in range(T):
            frame = x[:,:,t,:,:]
            frame_feature = self.dinov2(frame)
            if t==0:
                frame_features = frame_feature.unsqueeze(-1)
            else:
                frame_features = torch.cat([frame_features, frame_feature.unsqueeze(-1)], dim=-1)

        frame_inner_features = self.iffa(frame_features.permute(0,2,1),self.adj)
        frame_inner_features = frame_inner_features.permute(0,2,1)
        print(frame_inner_features.shape)
        print(frame_features.shape)
        spatiotemporal_features = self.spfa_conv(frame_features.unsqueeze(1))
        spatiotemporal_features = self.spfa_pooling(spatiotemporal_features)
        print(spatiotemporal_features.shape)
        inner_features = self.inner_pooling(frame_inner_features.unsqueeze(1))
        print(inner_features.shape)

        concat_feature = torch.cat([spatiotemporal_features[:,0,:,0],inner_features[:,0,:,0]],1)
        #print(concat_feature.shape)
        output = self.cls(concat_feature)
        
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora_depth', type=int, default=2, help='number of tuned attention blocks')
    args = parser.parse_args()
    model = dinov2_lora_GAT_MER_3cls(lora_depth=args.lora_depth).cuda()

    x = torch.ones(2,3,8,224,224).cuda()
    y = model(x)
    print(y.shape)
                
    