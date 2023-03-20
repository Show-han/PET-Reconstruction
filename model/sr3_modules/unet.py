import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()

        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
                noise_level_emb_dim, dim_out, use_affine_level)


        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        if time_emb is not None:
            h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class GuidedNet(nn.Module):
    def __init__(self, in_dim, down_scale, in_channel=8, inner_channel=16, noise_level_channel=None, dropout=0, norm_groups=8, channel_mults=(1,2,3,4)):
        super().__init__()
        self.feature = None
        self.downs = nn.Sequential(
            Downsample(in_dim),
        )
        self.loss_func = nn.L1Loss(reduction='sum')
        if down_scale > 1:
            for i in range (down_scale-1):
                self.downs.add_module(str(i+1),Downsample(in_dim))
        block = [nn.Conv2d(in_dim, in_channel, 3, padding=1)]

        pre_channel = in_channel
        num_mults = len(channel_mults)
        for ind in range(num_mults):
            channel_mult = inner_channel * channel_mults[ind]
            block.append(ResnetBlocWithAttn(
                pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                dropout=dropout, with_attn=False))
            pre_channel = channel_mult
        self.conv = nn.Conv2d(pre_channel, in_dim, 3, padding=1)
        self.block = nn.ModuleList(block)
    def forward(self, x, high,t):
        x = self.downs(x)
        b, c, h, w = x.shape
        high = self.downs(high)
        for layer in self.block:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        # x = self.block(x)
        self.feature = x
        x = self.conv(x)
        l_loss = self.loss_func(x, high)/int(b*c*h*w)
        return x ,l_loss

    def get_feature(self):
        return self.feature

class GuidedResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, guide_dim, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        if noise_level_emb_dim is not None:
            self.noise_func = FeatureWiseAffine(
                noise_level_emb_dim, dim_out, use_affine_level)


        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.conv1 = nn.Conv2d(guide_dim, dim, 1)
        self.conv2 = nn.Conv2d(guide_dim, dim_out, 1)
        self.conv_aff1 = nn.Conv2d(dim * 3, dim, 1)
        self.conv_aff2 = nn.Conv2d(dim_out * 3, dim_out, 1)

    def forward(self, x, time_emb, ax_feature, fr_feature):
        b, c, h, w = x.shape
        b_ax, c_ax, h_ax, w_ax = ax_feature.shape
        b_fr, c_fr, h_fr, w_fr = fr_feature.shape

        assert h_ax == h_fr == h
        assert w_ax == w_fr  == w
        new_ax_feature = ax_feature
        new_fr_feature = fr_feature

        ax_feature = self.conv1(new_ax_feature)
        fr_feature = self.conv1(new_fr_feature)

        h = torch.cat([x, ax_feature, fr_feature], dim=1)
        h = self.conv_aff1(h)
        h = self.block1(h)
        if time_emb is not None:
            h = self.noise_func(h, time_emb)
        ax_feature = self.conv2(new_ax_feature)
        fr_feature = self.conv2(new_fr_feature)

        h = torch.cat([h, ax_feature, fr_feature], dim=1)
        h = self.conv_aff2(h)
        h = self.block2(h)
        return h + self.res_conv(x)

class GuidedResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, guide_dim, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = GuidedResnetBlock(
            dim, dim_out, guide_dim, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb, ax_feature, fr_feature):
        x = self.res_block(x, time_emb, ax_feature, fr_feature)
        if(self.with_attn):
            x = self.attn(x)
        return x



class UNet(nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel=1,
        inner_channel=64,
        norm_groups=32,
        channel_mults=(1, 2, 3, 4),
        attn_res=(8,),
        res_blocks=2,
        dropout=0,
        with_noise_level_emb=True,
        image_size=128,
        guide_res=(64, 32),
        guide_dim=(64, 64, 64)
    ):
        super().__init__()
        self.res_blocks = res_blocks

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                if now_res in guide_res:
                    downs.append(GuidedResnetBlocWithAttn(
                        pre_channel, channel_mult, guide_dim=guide_dim[guide_res.index(now_res)], noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                else:
                    downs.append(ResnetBlocWithAttn(
                        pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            GuidedResnetBlocWithAttn(pre_channel, pre_channel, guide_dim=guide_dim[-1], noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            GuidedResnetBlocWithAttn(pre_channel, pre_channel, guide_dim=guide_dim[-1], noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time, ax_feature = [], fr_feature = []):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        index = 0
        cnt = 0

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            elif isinstance(layer, GuidedResnetBlocWithAttn):
                x = layer(x, t, ax_feature[index], fr_feature[index])
                cnt += 1
                if cnt == self.res_blocks:
                    index += 1
                    cnt = 0
            else:
                x = layer(x)
            feats.append(x)
        for layer in self.mid:
            if isinstance(layer, GuidedResnetBlocWithAttn):
                x = layer(x, t, ax_feature[index], fr_feature[index])
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)
        return self.final_conv(x)

if __name__ == "__main__":
    #model = GuidedNet(in_dim=2, down_scale=3, in_channel=8, inner_channel=16, noise_level_channel=None, dropout=0, norm_groups=8, channel_mults=(1,2,3,4)).to("cuda")

    model = UNet(with_noise_level_emb=False).to("cuda")
    noise_level = torch.FloatTensor(
        [0.5]).repeat(1, 1).to("cuda")
    sample = torch.randn(2, 1, 128, 128).to("cuda")
    x = [torch.randn(2, 64, 64, 64).to("cuda"),torch.randn(2,2*32,32,32).to("cuda"),torch.randn(2,2*32,16,16).to("cuda"),torch.randn(2,2*32,16,16).to("cuda")]
    y=model(sample, noise_level, x, x)
    # x = torch.randn(1,2,64,64).to("cuda")
    # a = torch.randn(4,1,5,64,64).to("cuda")
    # b = torch.randn(4,1,1,64,64).to("cuda")
    # c = torch.randn(4,1,1,64,64).to("cuda")
    # y = model(x, noise_level,a,b,c)
    # y ,loss= model(sample , sample,None)
    print(y.shape)
    # print(model.get_feature().shape)
    # print(('1' in ['1']))