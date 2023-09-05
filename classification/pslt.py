# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from module import SepConv2d
import torch
import torch.nn as nn
import math
from torch import einsum, sqrt
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from einops import rearrange
from module import SepConv2d
import math
# from mmcv_custom import load_checkpoint
import os
# from visualizer import get_local


class ChannelSELayer_SA(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer_SA, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        # batch_size, L, num_channels = input_tensor.size()
        # H, W = int(L ** 0.5), int(L ** 0.5)

        # inp = input_tensor.permute(0, 2, 1).view(batch_size, num_channels, H, W)
        # Average along each channel
        # squeeze_tensor = inp.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        # print(squeeze_tensor.shape)
        fc_out_1 = self.relu(self.fc1(input_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        # a, _, b = input_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2)
        return output_tensor


class MV2Block_SE(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.GELU(),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        self.se = ChannelSELayer(oup)

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        if self.use_res_connect:
            return x + out
        else:
            return out

class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class Mlp_Light(nn.Module):
    def __init__(self, in_features, input_resolution, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., reduce_scale=4, **kargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(hidden_features/reduce_scale)
        self.input_resolution = input_resolution
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, 1),
            act_layer(),
            nn.BatchNorm2d(hidden_features),
            nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features),
            act_layer(),
            nn.BatchNorm2d(hidden_features),
            nn.Conv2d(hidden_features, out_features, 1, 1),
            nn.BatchNorm2d(out_features),
        )
        
        
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert H*W == L , "input feature has wrong size"

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(B, H*W, C)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# @get_local('attention_map')
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., k_conv_value=1, 
                 kv_down_scale=0, input_v=False, input_kv=False, v2k=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.k_conv_value = k_conv_value
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        if input_v:
            self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        elif input_kv:
            print('input k v from last branch')
            self.x2q = nn.Linear(dim, dim, bias=qkv_bias)
            if v2k:
                print('v2k')
                self.v2k = nn.Linear(dim, dim, bias=qkv_bias)
            else:
                self.v2k = None
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None, in_v=None, in_kv=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        H, W = self.window_size
        
        if in_v is not None:
            shortcut = x
            # v = self.q2q(in_v)    # have q2q or not
            v = in_v.reshape(B_, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
            # v = v.reshape(B_, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k = kv[0], kv[1]
        else:
            qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # B_, head, N, d
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + shortcut if in_v is not None or in_kv is not None else x
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class multi_branch_LadderBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=[3,5,7], shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ch_shuffle=False, add_dim=False, 
                 query_branch=False, branch2input=False, pointwise=False, branch2input_conv=False,
                 pool_less_token=False, pool_same_token=False, max_out=False, out_dim=None, global_SA=False,
                 extra_conv=False, multi_shift=False, branch_attn=False, branch_add=False,
                 multi_shift_first2input=False, dilateConv_branch=False, sum_cat=False, branch_attn_cat=False,
                 Mlp=Mlp_Light, has_se=False, no_shift=False, has_se_fc=False, multi_pool=False, multi_conv=False, group_conv=False,
                 conv_v=False, conv_before_branch=False, no_mlp=False, pointwise_Light=False, kv_down_scale=0, input_v=False,
                 input_kv=False, v2k=False, has_avg_se=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_branch = len(window_size)
        self.blocks = []
        self.ch_shuffle = ch_shuffle
        self.add_dim = add_dim
        self.num_branch = self.num_branch + 1 if global_SA else self.num_branch
        self.num_branch = self.num_branch + 1 if extra_conv else self.num_branch
        if conv_before_branch:
            self.conv_before_branch = SepConv2d(dim, dim, 3, 1, 1)
        else:
            self.conv_before_branch = None
        dim = int(dim / self.num_branch)
        self.query_branch = query_branch
        self.branch2input = branch2input
        self.pointwise = pointwise
        self.pointwise_Light = pointwise_Light
        self.branch2input_conv = branch2input_conv
        self.max_out = max_out
        if branch2input_conv:
            self.branch_conv = []
        self.pool_less_token = pool_less_token
        self.pool_same_token = pool_same_token
        self.global_SA = global_SA
        self.extra_conv = extra_conv
        shift_direction = 1
        self.branch_attn = branch_attn
        self.branch_attn_cat = branch_attn_cat
        self.branch_add = branch_add
        self.multi_shift_first2input = multi_shift_first2input
        self.dilateConv_branch = dilateConv_branch
        self.sum_cat = sum_cat
        self.has_se = has_se
        self.has_se_fc = has_se_fc
        self.has_avg_se = has_avg_se
        self.group_conv = group_conv
        self.input_v = input_v
        self.input_kv = input_kv
        

        for i in range(self.num_branch):
            if multi_shift:
                if i == 0 or kv_down_scale > 0:
                    shift_size = 0
                elif i == 1:
                    shift_size = window_size[i] // 2
                    shift_direction = 1  # not assigned in multi_shift
                elif i == 2:
                    # shift_size = 0
                    # 3 left
                    shift_size = window_size[i] // 2
                    shift_direction = 0
                else:
                    print('not implement for more than 3 branches!!')
                    exit()

            elif no_shift:
                shift_size = 0
            else:
                shift_size = window_size[i] // 2 if shift_size == None else shift_size
            
            
            self.blocks.append(LadderSABlock(dim=dim, input_resolution=input_resolution,
                                num_heads=num_heads, shift_size=shift_size, mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                                drop_path=drop_path, norm_layer=norm_layer, out_dim=out_dim, 
                                shift_direction=shift_direction, Mlp=Mlp, no_mlp=no_mlp, 
                                kv_down_scale=kv_down_scale, input_v=input_v if i > 0 else False, 
                                input_kv=input_kv if i>0 else False, v2k=v2k))
        
        self.blocks = nn.ModuleList(self.blocks)

        if self.pointwise:
            dim_in = dim*self.num_branch
            self.pointconv = nn.Linear(dim_in, dim*self.num_branch)

        if self.has_se:
            self.se = ChannelSELayer_SA(dim*self.num_branch)
             

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # print(x.shape)
        # print(H)
        assert L == H * W, "input feature has wrong size"

        dim = int(C / self.num_branch)
        # print('B:{}, L:{}, C:{}'.format(B, L, C))
        for i in range(self.num_branch):
            # print(i)
            idx = 2 * i if self.add_dim else i
            x_input = x[:,:,dim*i:dim*(i+1)]
            if self.input_v:
                in_v = None if i == 0 else xs[:, :, dim*(i-1):dim*i]
            else:
                in_v = None
            
            x_ = self.blocks[idx](x_input, in_v)
            xs = torch.cat((xs, x_), dim=2) if i>0 else x_
        
        if self.has_se:
            xs = self.se(xs)
            
        shortcut = xs         
        
        if self.pointwise:
            xs = self.pointconv(xs)

        xs = xs + shortcut
        return xs

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class LadderSABlock(nn.Module):
    r""" Ladder Self-Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, out_dim=None, shift_direction=1, Mlp=Mlp_Light,
                 pool_size=0, conv_size=0, k_conv_value=1, no_mlp=False, kv_down_scale=0,
                 input_v=False, input_kv=False, v2k=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.shift_direction = shift_direction
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        if conv_size > 0:
            self.conv = nn.Sequential(
                SepConv2d(dim, dim, conv_size, 1, conv_size//2),
                act_layer(),
                nn.BatchNorm2d(dim),
            )
        else:
            self.conv = None

        if pool_size > 0:
            self.pool = nn.AvgPool2d(pool_size, 1, pool_size//2)
        else:
            self.pool = None

        attn_func = WindowAttention

        self.attn = attn_func(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            k_conv_value=k_conv_value, input_v=input_v, input_kv=input_kv, v2k=v2k)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if no_mlp:
            self.mlp = None
        else:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, input_resolution=input_resolution, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0 and conv_size == 0 and kv_down_scale == 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            if self.shift_direction == 1:
                h_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size),
                            slice(-self.window_size, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
            else:
                h_slices = (slice(0, self.shift_size),
                            slice(self.shift_size, self.window_size),
                            slice(self.window_size, None))
                w_slices = (slice(0, self.shift_size),
                            slice(self.shift_size, self.window_size),
                            slice(self.window_size, None))
                cnt = 8
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt = cnt -1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, in_v=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        # for less token
        # H, W = int(math.sqrt(L)), int(math.sqrt(L))
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)
        
        if in_v is not None:
            in_v = in_v.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if self.shift_direction == 1:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                if in_v is not None:
                    shift_in_v = torch.roll(in_v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                
            else:
                shifted_x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                if in_v is not None:
                    shift_in_v = torch.roll(in_v, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                
        else:
            shifted_x = x
            if in_v is not None:
                shift_in_v = in_v
            

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C   #window_partition
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        if in_v is not None:
            in_v_windows = window_partition(shift_in_v, self.window_size)  # nW*B, window_size, window_size, C   #window_partition
            in_v_windows = in_v_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        else:
            in_v_windows = None


        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask, in_v=in_v_windows)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C   # H, W
        if in_v_windows is not None:
            in_v_windows = in_v_windows.view(-1, self.window_size, self.window_size, C)
            shift_in_v = window_reverse(in_v_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            if self.shift_direction == 1:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
                if in_v is not None:
                    in_v = torch.roll(shift_in_v, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = torch.roll(shifted_x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                if in_v is not None:
                    in_v = torch.roll(shift_in_v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # print(x.shape)
        x = x.view(B, H*W, C)

        x = shortcut + self.drop_path(x)

        if self.mlp is not None:
            x = x + self.drop_path(self.mlp(self.norm2(x)))   # after 12-25
            
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer_multi_branch(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size=[7,7,7],
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 ch_shuffle=False, add_dim=False, query_branch=False, branch2input=False,
                 pointwise=False, branch2input_conv=False, pool_same_token=False, pool_less_token=False,
                 max_out=False, aggressive_dim=False, shift_window=True, global_SA=False,
                 extra_conv=False, multi_shift=False, branch_attn=False, branch_add=False,
                 multi_shift_first2input=False, dilateConv_branch=False, sum_cat=False,
                 branch_attn_cat=False, Mlp=Mlp_Light, has_se=False, no_shift=False, has_se_fc=False,
                 multi_pool=False, multi_conv=False, group_conv=False, conv_v=False,
                 conv_before_branch=False, no_mlp=False, pointwise_Light=False,
                 kv_down_scale=0, input_v=False, input_kv=False,
                 v2k=False, has_avg_se=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.aggressive_dim = aggressive_dim

        # build blocks
        self.blocks = []
        

        self.blocks = nn.ModuleList(self.blocks)
        self.blocks = nn.ModuleList([
            multi_branch_LadderBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    shift_size=0 if (i % 2 == 0) or not shift_window else None,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                    norm_layer=norm_layer, ch_shuffle=ch_shuffle, add_dim=add_dim, 
                                    query_branch=query_branch, branch2input=branch2input, pointwise=pointwise,
                                    branch2input_conv=branch2input_conv,pool_same_token=pool_same_token, pool_less_token=pool_less_token,
                                    max_out=max_out, global_SA=global_SA, extra_conv=extra_conv, multi_shift=multi_shift, branch_attn=branch_attn,
                                    branch_add=branch_add, multi_shift_first2input=multi_shift_first2input,
                                    dilateConv_branch=dilateConv_branch, sum_cat=sum_cat, branch_attn_cat=branch_attn_cat, Mlp=Mlp, has_se=has_se,
                                    no_shift=no_shift, has_se_fc=has_se_fc, multi_pool=multi_pool, multi_conv=multi_conv, group_conv=group_conv,
                                    conv_v=conv_v, conv_before_branch=conv_before_branch, no_mlp=no_mlp, pointwise_Light=pointwise_Light,
                                    kv_down_scale=kv_down_scale, input_v=input_v, input_kv=input_kv,
                                    v2k=v2k, has_avg_se=has_avg_se)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"



class PatchEmbed_stem_3conv(nn.Module):
    def __init__(self, img_size=224, patch_size=2, in_chans=3, embed_dim=96, norm_layer=None, act_layer=nn.GELU):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        
        self.proj = nn.Sequential(
            nn.Conv2d(3, self.embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False),
            act_layer(),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim, kernel_size=3, stride=1, padding=1, bias=False),

            act_layer(),
            nn.BatchNorm2d(self.embed_dim),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            act_layer(),
            nn.BatchNorm2d(self.embed_dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x



class PSLTransformer(nn.Module):
    r""" PSLT Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=2, in_chans=3, num_classes=1000,
                 embed_dim=36, conv_depths=[3, 3], depths=[10, 3], num_heads=[4, 8],
                 window_size=[7, 7, 7], mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, PatchMerging=PatchMerging,
                 use_checkpoint=False, BasicLayer=BasicLayer_multi_branch, PatchEmbed=PatchEmbed_stem_3conv, 
                 Mlp=Mlp_Light, multi_shift=False, shift_window=True, has_se=False, input_v=False, 
                 pointwise=False, distilled=False, distill_fc=False,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.distilled = distilled

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.num_conv_layers = len(conv_depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers + self.num_conv_layers))

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.conv_layers = nn.ModuleList()
        for t, depth in enumerate(conv_depths):
            for j in range(depth):
                if j == 0:
                    conv_layer = MV2Block_SE(int(embed_dim * 2 ** t), int(embed_dim * 2 ** (t+1)), stride=2)
                else:
                    conv_layer = MV2Block_SE(int(embed_dim * 2 ** (t+1)), int(embed_dim * 2 ** (t+1)))
                self.conv_layers.append(conv_layer)

        # MV2 for downsample
        conv_layer = MV2Block_SE(int(embed_dim * 2 ** self.num_conv_layers), int(embed_dim * 2 ** (self.num_conv_layers+1)), stride=2)
        self.conv_layers.append(conv_layer)
        


        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** (i_layer+self.num_conv_layers+1)),
                               input_resolution=(patches_resolution[0] // (2 ** (i_layer+self.num_conv_layers+1)),
                                                 patches_resolution[1] // (2 ** (i_layer+self.num_conv_layers+1))),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pointwise=pointwise, shift_window=shift_window, 
                               multi_shift=multi_shift,
                               Mlp=Mlp, has_se=has_se, 
                               input_v=input_v,
                               )
            self.layers.append(layer)
                        

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if self.distilled:
            self.head_dist = nn.Linear(self.num_features, num_classes) if distill_fc else nn.Identity()

        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.flatten(2).transpose(1, 2)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1

        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x_fc = self.head(x)
        if self.distilled:
            x_dist = self.head_dist(x)
            if self.training:
                return x_fc, x_dist
            else:
                return (x_fc + x_dist) / 2
        else:
            return x_fc


@register_model
def PSLT(pretrained=False, **kwargs):
    model = PSLTransformer(img_size=224, conv_depths=[3, 3], patch_norm=False, window_size=[7,7,7], depths=[10, 3], num_heads=[4, 8],
                           embed_dim=36, drop_path_rate=0.2, num_classes=1000, BasicLayer=BasicLayer_multi_branch, pointwise=True, 
                           PatchEmbed=PatchEmbed_stem_3conv, Mlp=Mlp_Light, multi_shift=True, has_se=True, input_v=True,
                           distilled=False, distill_fc=False)
    return model

@register_model
def PSLT_large(pretrained=False, **kwargs):
    model = PSLTransformer(img_size=224, conv_depths=[3, 3], patch_norm=False, window_size=[7,7,7], depths=[10, 3], num_heads=[4, 8],
                           embed_dim=48, drop_path_rate=0.2, num_classes=1000, BasicLayer=BasicLayer_multi_branch, pointwise=True, 
                           PatchEmbed=PatchEmbed_stem_3conv, Mlp=Mlp_Light, multi_shift=True, has_se=True, input_v=True,
                           distilled=False, distill_fc=False)
    return model


@register_model
def PSLT_tiny(pretrained=False, **kwargs):
    model = PSLTransformer(img_size=224, conv_depths=[3, 3], patch_norm=False, window_size=[7,7,7], depths=[10, 3], num_heads=[4, 8],
                           embed_dim=24, drop_path_rate=0.2, num_classes=1000, BasicLayer=BasicLayer_multi_branch, pointwise=True,
                           PatchEmbed=PatchEmbed_stem_3conv, Mlp=Mlp_Light, multi_shift=True, has_se=True, input_v=True,
                           distilled=False, distill_fc=False)
    return model
