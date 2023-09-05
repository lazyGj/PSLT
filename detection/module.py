import torch
from torch import mode, nn, einsum
from einops import rearrange


class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pointwise(x)
        return x

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class ConvAttention_wo_cls(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 distilled=False, dw_scale=1):

        super().__init__()
        self.distilled = distilled
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)    # b, 2+n, #head * dim
        return out


class ConvAttention(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False, distilled=False, dw_scale=1):

        super().__init__()
        self.last_stage = last_stage
        self.distilled = distilled
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h) 
            # b batch; h head; n #tokens=1; d #dimensions 
            if self.distilled:
                dis_token = x[:, 1]
                dis_token = rearrange(dis_token.unsqueeze(1), 'b n (h d) -> b h n d', h=h)
                x = x[:, 2:]
            else:
                x = x[:, 1:]
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            if self.distilled:
                q = torch.cat((cls_token, dis_token, q), dim=2)
                v = torch.cat((cls_token, dis_token, v), dim=2)
                k = torch.cat((cls_token, dis_token, k), dim=2)
            else:
                # print(cls_token.shape)
                # print(q.shape)
                q = torch.cat((cls_token, q), dim=2)
                v = torch.cat((cls_token, v), dim=2)
                k = torch.cat((cls_token, k), dim=2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)    # b, 2+n, #head * dim
        return out

class ConvAttention_spa_att(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False, distilled=False):

        super().__init__()
        self.last_stage = last_stage
        self.distilled = distilled
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = torch.nn.Conv2d(2, 2*heads, kernel_size, q_stride, pad)
        self.to_k = torch.nn.Conv2d(2, 2*heads, kernel_size, k_stride, pad)
        # self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)
        self.to_v = torch.nn.Conv2d(2, 2*heads, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(2*heads, dim),   # (inner_dim, dim)
            nn.Dropout(dropout)
        )# if project_out else nn.Identity()

        if last_stage:
            self.linear = nn.Linear(dim_head, 2)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h) 
            cls_token_attn = self.linear(cls_token)
            # b batch; h head; n #tokens=1; d #dimensions 
            if self.distilled:
                dis_token = x[:, 1]
                dis_token = rearrange(dis_token.unsqueeze(1), 'b n (h d) -> b h n d', h=h)
                x = x[:, 2:]
            else:
                x = x[:, 1:]
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        mean_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        x_attn = torch.cat([mean_x, max_x], dim=1)
        q = self.to_q(x_attn)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x_attn)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x_attn)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            if self.distilled:
                q = torch.cat((cls_token, dis_token, q), dim=2)
                v = torch.cat((cls_token, dis_token, v), dim=2)
                k = torch.cat((cls_token, dis_token, k), dim=2)
            else:
                q = torch.cat((cls_token_attn, q), dim=2)
                v = torch.cat((cls_token_attn, v), dim=2)
                k = torch.cat((cls_token_attn, k), dim=2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        # print(v.shape)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # print(out.shape)
        out =  self.to_out(out)    # b, 2+n, #head * dim
        # out = rearrange(out, 'b n d -> b d n')
        # print(out.shape)
        return out




class ConvAttention_spa_att_less_token(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False, distilled=False):

        super().__init__()
        self.last_stage = last_stage
        self.distilled = distilled
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = torch.nn.Conv2d(2, 2*heads, kernel_size, q_stride, pad)
        self.to_k = torch.nn.Conv2d(2, 2*heads, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)
        # self.to_v = torch.nn.Conv2d(2, 2*heads, kernel_size, v_stride, pad)
        self.avg_pool = nn.AvgPool2d(3, 2, 1)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bicubic')

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),   # (inner_dim, dim)
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if last_stage:
            self.linear = nn.Linear(dim_head, 2)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h) 
            cls_token_attn = self.linear(cls_token)
            # b batch; h head; n #tokens=1; d #dimensions 
            if self.distilled:
                dis_token = x[:, 1]
                dis_token = rearrange(dis_token.unsqueeze(1), 'b n (h d) -> b h n d', h=h)
                x = x[:, 2:]
            else:
                x = x[:, 1:]
        
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        x_pool = self.avg_pool(x)
        mean_x = torch.mean(x_pool, dim=1, keepdim=True)
        max_x, _ = torch.max(x_pool, dim=1, keepdim=True)
        x_attn = torch.cat([mean_x, max_x], dim=1)
        q = self.to_q(x_attn)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x_pool)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x_attn)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            if self.distilled:
                q = torch.cat((cls_token, dis_token, q), dim=2)
                v = torch.cat((cls_token, dis_token, v), dim=2)
                k = torch.cat((cls_token, dis_token, k), dim=2)
            else:
                q = torch.cat((cls_token_attn, q), dim=2)
                v = torch.cat((cls_token, v), dim=2)
                k = torch.cat((cls_token_attn, k), dim=2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)    # b, 2+n, #head * dim
        if self.last_stage:
            if self.distilled:
                out_pool = out[:, 2:]
            else:
                out_pool = out[:, 1:]
            out_pool = rearrange(out_pool, 'b (l w) d -> b d l w', l=int(self.img_size/2), w=int(self.img_size/2))
            out_up = self.up_sample(out_pool)
            out_up = rearrange(out_up, 'b d l w -> b (l w) d', l=self.img_size, w=self.img_size)
            if self.distilled:
                out = torch.cat((out[:,0].unsqueeze(1), out[:,1].unsqueeze(1), out_up), dim=1)
            else:
                out = torch.cat((out[:,0].unsqueeze(1), out_up), dim=1)
        else:
            out = rearrange(out, 'b (l w) d -> b d l w', l=int(self.img_size/2), w=int(self.img_size/2))
            out = self.up_sample(out)
            out = rearrange(out, 'b d l w -> b (l w) d', l=self.img_size, w=self.img_size)
        return out


class ConvAttention_less_token(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False, distilled=False, dw_scale=2):

        super().__init__()
        self.last_stage = last_stage
        self.distilled = distilled
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.dw_scale = dw_scale
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)
        dw_kernel = 3 if dw_scale == 2 else 7
        dw_stride = 2 if dw_scale == 2 else 4
        dw_pad = 1 if dw_scale ==2 else 2
        self.avg_pool = nn.AvgPool2d(dw_kernel, dw_stride, dw_pad)
        # self.up_sample = nn.Upsample(scale_factor=dw_scale, mode='bicubic')
        self.up_sample = nn.Upsample(size=[self.img_size, self.img_size], mode='bicubic')

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        if self.last_stage:
            cls_token = x[:, 0]
            cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h) 
            # b batch; h head; n #tokens=1; d #dimensions 
            if self.distilled:
                dis_token = x[:, 1]
                dis_token = rearrange(dis_token.unsqueeze(1), 'b n (h d) -> b h n d', h=h)
                x = x[:, 2:]
            else:
                x = x[:, 1:]
        
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)
        x_pool = self.avg_pool(x)
        q = self.to_q(x_pool)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x_pool)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x_pool)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        if self.last_stage:
            if self.distilled:
                q = torch.cat((cls_token, dis_token, q), dim=2)
                v = torch.cat((cls_token, dis_token, v), dim=2)
                k = torch.cat((cls_token, dis_token, k), dim=2)
            else:
                q = torch.cat((cls_token, q), dim=2)
                v = torch.cat((cls_token, v), dim=2)
                k = torch.cat((cls_token, k), dim=2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)    # b, 2+n, #head * dim
        if self.last_stage:
            if self.distilled:
                out_pool = out[:, 2:]
            else:
                out_pool = out[:, 1:]
            out_pool = rearrange(out_pool, 'b (l w) d -> b d l w', l=int(self.img_size/self.dw_scale), w=int(self.img_size/self.dw_scale))
            out_up = self.up_sample(out_pool)
            out_up = rearrange(out_up, 'b d l w -> b (l w) d', l=self.img_size, w=self.img_size)
            if self.distilled:
                out = torch.cat((out[:,0].unsqueeze(1), out[:,1].unsqueeze(1), out_up), dim=1)
            else:
                out = torch.cat((out[:,0].unsqueeze(1), out_up), dim=1)
        else:
            out = rearrange(out, 'b (l w) d -> b d l w', l=int(self.img_size/self.dw_scale), w=int(self.img_size/self.dw_scale))
            out = self.up_sample(out)
            out = rearrange(out, 'b d l w -> b (l w) d', l=self.img_size, w=self.img_size)
        return out


class ConvAttention_mask(nn.Module):
    def __init__(self, dim, img_size, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False, distilled=False):

        super().__init__()
        self.last_stage = last_stage
        self.distilled = distilled
        self.img_size = img_size
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)

        self.to_q_mask = torch.nn.Conv2d(2, 2*heads, kernel_size, q_stride, pad)
        self.to_k_mask = torch.nn.Conv2d(2, 2*heads, kernel_size, k_stride, pad)
        # self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad)
        self.to_v_mask = torch.nn.Conv2d(2, 2*heads, kernel_size, v_stride, pad)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        # if self.last_stage:
        #     cls_token = x[:, 0]
        #     cls_token = rearrange(cls_token.unsqueeze(1), 'b n (h d) -> b h n d', h = h) 
        #     # b batch; h head; n #tokens=1; d #dimensions 
        #     if self.distilled:
        #         dis_token = x[:, 1]
        #         dis_token = rearrange(dis_token.unsqueeze(1), 'b n (h d) -> b h n d', h=h)
        #         x = x[:, 2:]
        #     else:
        #         x = x[:, 1:]
        x = rearrange(x, 'b (l w) n -> b n l w', l=self.img_size, w=self.img_size)

        mean_x = torch.mean(x, dim=1, keepdim=True)
        max_x, _ = torch.max(x, dim=1, keepdim=True)
        x_attn = torch.cat([mean_x, max_x], dim=1)
        q_mask = self.to_q_mask(x_attn)
        q_mask = rearrange(q_mask, 'b (h d) l w -> b h (l w) d', h=h)

        k_mask = self.to_k_mask(x_attn)
        k_mask = rearrange(k_mask, 'b (h d) l w -> b h (l w) d', h=h)
        dots_mask = einsum('b h i d, b h j d -> b h i j', q_mask, k_mask) * self.scale
        dots_mask = dots_mask.softmax(-1)
        _, k_index = dots_mask.topk(4, dim=-1)
        zeros = torch.zeros_like(dots_mask) - 1e5
        one_h = zeros.scatter_(-1, k_index, 1.0)
        hard_mask = one_h - dots_mask.detach() + dots_mask

        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        v = self.to_v(x)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(x)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)

        # if self.last_stage:
        #     if self.distilled:
        #         q = torch.cat((cls_token, dis_token, q), dim=2)
        #         v = torch.cat((cls_token, dis_token, v), dim=2)
        #         k = torch.cat((cls_token, dis_token, k), dim=2)
        #     else:
        #         q = torch.cat((cls_token, q), dim=2)
        #         v = torch.cat((cls_token, v), dim=2)
        #         k = torch.cat((cls_token, k), dim=2)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        dots = dots.mul(hard_mask)

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)    # b, 2+n, #head * dim
        return out