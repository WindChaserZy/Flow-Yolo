import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny
from nets.FlowNet import FlowNetS
from nets.FeatureEmbedding import FeatureEmbedding

class FeatureNet(nn.Module):

    K = 4

    def __init__(self, flow_net : FlowNetS, feature_embeddings : nn.ModuleList):
        super(FeatureNet, self).__init__()
        self.flow_net = flow_net
        self.feature_embeddings = feature_embeddings

    def forward(self, x, feature_maps_tuple):
        x = self._feature_aggregation(x, feature_maps_tuple)
        return x
    
    def _feature_aggregation(self, frames, feature_maps_tuple):
        device = next(self.parameters()).device
        frames = frames.to(device)
        # [tensor(N,C,H,W)]
        feature_maps_aggregation_list = []
        for feature_maps, feature_embedding in zip(feature_maps_tuple, self.feature_embeddings):
            # N, C, H, W
            N, C, _, _ = feature_maps.shape
            f_i_aggregation_list = []
            for i in range(N):
                w_list = []
                f_list = []
                for j in range(max(0, i - FeatureNet.K), min(N, i + FeatureNet.K + 1)):
                    pad_frames = torch.cat([frames[j:j+1], frames[i:i+1]], dim=1)
                    flow_ji = self.flow_net(pad_frames)
                    # 1, c, h, w
                    f_ji = self._h_feature_warp(feature_maps[j:j+1], flow_ji, device)
                    # f_ji = feature_maps[j : j + 1]
                    # 1, emb
                    f_ji_emb, f_i_emb = feature_embedding(f_ji), feature_embedding(feature_maps[i:i+1])
                    # 1
                    w_ji = torch.exp(F.cosine_similarity(f_ji_emb, f_i_emb)).reshape(1,1,1,1)
                    # 1, c, 1, 1
                    w_ji.repeat(1, C, 1, 1)
                    f_list.append(f_ji)
                    w_list.append(w_ji)
                # 2K, c, h, w
                f = torch.concatenate(f_list, dim=0)
                # 2K, c, 1, 1
                w = torch.concatenate(w_list, dim=0)
                # 1, c, h, w
                f_i_aggregation = torch.sum(f * w / torch.sum(w), dim = 0, keepdim=True)
                f_i_aggregation_list.append(f_i_aggregation)
            feature_maps_aggregation_list.append(torch.concatenate(f_i_aggregation_list))
        return tuple(feature_maps_aggregation_list)


    # predict i th frame's feature map from k th and flow k->i
    def _feature_warp (self, f_k : torch.Tensor, flow : torch.Tensor, device : str):
        n, c, h, w = f_k.shape
        kernel_size = 2
        f_i = torch.zeros_like(f_k)
        flo = - F.interpolate(flow, size=(h,w), mode='bilinear', align_corners=False)

        for px in range(w):
            for py in range(h):
                dpx = flo[:, 0:1, py, px]
                dpy = flo[:, 1:, py, px]
                i, j = torch.floor(py + dpy), torch.floor(px + dpx)
                di, dj = py + dpy - i, px + dpx - j
                G = torch.concat([di * dj, di * (1 - dj), (1 - di) * dj, (1 - di) * (1 - dj)], dim=1).reshape(n, 1, kernel_size, kernel_size)
                # n, c, kernel, kernel
                G = G.repeat(1, c, 1, 1).to(device)
                grid = torch.zeros(n, kernel_size, kernel_size, 2).to(device)
                for gy in range(kernel_size):
                    for gx in range(kernel_size):
                        grid[:, gy, gx, 0:1] = 2 * (j + gx) / (w - 1) - 1
                        grid[:, gy, gx, 1:] = 2 * (i + gy) / (h - 1) - 1
                # n, c, kernel, kernel
                patch = F.grid_sample(f_k, grid,  mode='bilinear', padding_mode='zeros', align_corners=True)
                f_i[:,:, py, px] = torch.sum(G * patch, dim=(2, 3))

        return f_i

    def _h_feature_warp(self, f_k : torch.Tensor, flow : torch.Tensor, device : str):
        n, c, h, w = f_k.shape
        kernel_size = 2
        f_i = torch.zeros_like(f_k).to(device)
        # n, 2, h, w
        flo = - F.interpolate(flow, size=(h,w), mode='bilinear', align_corners=False).to(device)

        # n, 2, h, w
        grid = torch.stack(torch.meshgrid(torch.Tensor(range(w)), torch.Tensor(range(h)), indexing='xy'), dim=0).squeeze(0).to(device)
        grid.repeat(n, 1, 1, 1)
        
        grid_xy = torch.floor(grid + flo)
        grid_dxy = grid + flo - grid_xy

        # n, c, 4, h, w
        G = torch.concat([grid_dxy[:,0:1] * grid_dxy[:,1:],       (1 - grid_dxy[:,0:1]) * grid_dxy[:,1:], 
                          grid_dxy[:,0:1] * (1 - grid_dxy[:,1:]), (1 - grid_dxy[:,0:1]) * (1 - grid_dxy[:,1:])],
                          dim=1).unsqueeze(1).repeat(1,c,1,1,1).to(device)
        
        grid_xy = grid_xy.permute(0,2,3,1)
        patch_list = []
        for y in range(2):
            for x in range(2):
                #n, h, w, 2
                sample_grid = torch.concat([2 * (grid_xy[:,:,:,0:1] + x) / (w - 1) - 1,
                                            2 * (grid_xy[:,:,:,1:] + y) / (h - 1) - 1], 
                                            dim = -1).to(device)
                #n, c, h, w
                sample_patch = F.grid_sample(f_k, sample_grid, mode='bilinear', padding_mode='zeros', align_corners=True)
                patch_list.append(sample_patch)
        # n, c, 4, h, w
        patch = torch.stack(patch_list, dim = 2)
        f_i = torch.sum(G * patch, dim=2)

        return f_i




#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=True, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone = CSPDarknet(base_channels, base_depth, phi, pretrained)
        flow_net = FlowNetS(batchNorm=False, pretrained=pretrained)
        feature_embeddings = nn.ModuleList([FeatureEmbedding(128, 2048), FeatureEmbedding(256, 2048), FeatureEmbedding(512, 2048)])
        self.feature_net = FeatureNet(flow_net, feature_embeddings)
            
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, x):
        #  backbone
        feature_tuple = self.backbone(x)
        feat1, feat2, feat3 = self.feature_net(x, feature_tuple)

        # 20, 20, 1024 -> 20, 20, 512
        P5          = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4          = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4          = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3          = self.conv3_for_upsample2(P3)
        
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2

