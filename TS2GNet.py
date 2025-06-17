# 完整代码，仅修改模块类名，保持逻辑与内容完全不变

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# —— Asymmetric Conv Block (ACB) —— #
class ACB(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv_h = nn.Conv2d(channels, channels, (1, kernel_size),
                                padding=(0, kernel_size//2), groups=channels)
        self.conv_v = nn.Conv2d(channels, channels, (kernel_size, 1),
                                padding=(kernel_size//2, 0), groups=channels)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.conv_h(x) + self.conv_v(x)))

# —— Layerwise Scaling Residual Connection (LSRC) —— #
class LSRC(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x: [B, N, C]
        y = self.global_pool(x.permute(0,2,1)).squeeze(-1)  # [B, C]
        scale = self.fc(y).unsqueeze(1)                     # [B, 1, C]
        return x * scale.expand_as(x)

# —— Slice‐wise Enhanced Learning (SEL) —— #
class SEL(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.group_weights = nn.Parameter(torch.ones(in_channels))
        self.suppression_weights = nn.Parameter(torch.ones(in_channels))
        self.softmax = nn.Softmax(dim=0)
    def forward(self, *inputs):
        gw = self.softmax(self.group_weights)
        sw = self.softmax(self.suppression_weights)
        return sum(w * inp * (1 - s)
                   for w, inp, s in zip(gw, inputs, sw))

# —— Dynamic Spatial Attention (DSA) —— #
class DSA(nn.Module):
    def __init__(self, channel, patch_size):
        super().__init__()
        self.attn_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=2, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, 1, bias=False),
        )
        self.up_sample = nn.Upsample(mode='nearest',
                                     size=(channel, patch_size, patch_size))
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False),
            nn.Conv2d(channel, channel, 1, bias=False),
        )
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.attn_conv(x)
        y = self.up_sample(y.unsqueeze(1)).squeeze(1)
        y = self.conv(y)
        y = self.bn(y)
        y = self.sigmoid(y)
        return x * y + x, y

# —— Multiattention Module (MAM) —— #
class MAM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        f = 2
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.attention_embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//4, 1, bias=False),
            nn.BatchNorm2d(dim//4),
            nn.ReLU(),
            nn.Conv2d(dim//4, f*f*dim, 1, bias=False)
        )
        self.coeff1 = nn.Parameter(torch.tensor(1.0))
        self.coeff2 = nn.Parameter(torch.tensor(1.0))
        self.factor = f
    def forward(self, x):
        B, C, H, W = x.shape
        k1 = self.key_embed(x)
        v  = self.value_embed(x).view(B, C, -1)
        y  = torch.cat([k1, x], dim=1)
        att= self.attention_embed(y)
        att= att.view(B, C, self.factor*self.factor, H, W).mean(2).view(B, C, -1)
        k2 = (F.softmax(att, dim=-1) * v).view(B, C, H, W)
        return self.coeff1 * x + self.coeff2 * k2

# —— MS2F: Multi‐Scale Spectral–Spatial Fusion —— #
class MS2F(nn.Module):
    def __init__(self, channelNumber, patch_size):
        super().__init__()
        channels = (20,30,20,20,20,41)
        self.mgca = nn.ModuleList([MAM(d) for d in channels])
        self.mgsa = nn.ModuleList([DSA(ch, patch_size) for ch in channels])
        self.fuse_ca = nn.ModuleList([SEL(ch) for ch in channels])
        self.fuse_sa = nn.ModuleList([SEL(ch) for ch in channels])
        self.fusion_layer = nn.Sequential(
            nn.Conv2d(151, 453, 1, bias=False),
            nn.BatchNorm2d(453),
            nn.ReLU(),
            nn.Conv2d(453, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
    def forward(self, x):
        splits = [
            x[:, 0:20], x[:, 20:50], x[:, 50:70],
            x[:, 70:90], x[:, 90:110], x[:, 110:151]
        ]
        ca_outs, sa_outs = [], []
        for ca_block, sa_block, xi in zip(self.mgca, self.mgsa, splits):
            ca_outs.append(ca_block(xi))
            sa_outs.append(sa_block(xi)[0])
        fused = [f_ca(ca_o, sa_o)
                 for f_ca, ca_o, sa_o in zip(self.fuse_ca, ca_outs, sa_outs)]
        out = torch.cat(fused, dim=1)
        return self.fusion_layer(out)

# —— Multi‐Head Spectral Attention (MHSA) —— #
class MHSA(nn.Module):
    def __init__(self, dim, heads=4, dim_head=96, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim_head*heads*3, bias=False)
        self.attn   = nn.Softmax(dim=-1)
        self.scale  = dim_head ** -0.5
        self.to_out = nn.Sequential(nn.Linear(dim_head*heads, dim),
                                    nn.Dropout(dropout))
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q,k,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots  = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn  = self.attn(dots)
        out   = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out   = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# —— LRTM: Lightweight Residual Transformer Module —— #
class LRTM(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                MHSA(dim, heads, dim_head, dropout),
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                ACB(mlp_dim),
                nn.Linear(mlp_dim, dim), nn.Dropout(dropout),
                LSRC(dim)
            ]))
    def forward(self, x):
        for ln1, attn, ln2, fc1, gelu, do1, acb, fc2, do2, lsrc in self.layers:
            y = attn(ln1(x)); x = lsrc(x + y)
            y = fc1(ln2(x)); y = gelu(y); y = do1(y)
            B,N,C = y.shape
            y = y.permute(0,2,1).unsqueeze(2)
            y = acb(y).squeeze(2).permute(0,2,1)
            y = fc2(y); y = do2(y)
            x = lsrc(x + y)
        return x

# —— SE Block (Channel Squeeze‐Excitation) —— #
class SE(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# —— TS2FM: Temporal‐Spatial‐Spectral Fusion Module —— #
class TS2FM(nn.Module):
    def __init__(self, dim=256, heads=4, dropout=0.1, num_frames=3):
        super().__init__()
        self.num_frames = num_frames
        self.pos_emb = nn.Parameter(torch.randn(1, num_frames, dim))
        self.self_attn_p = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_attn_P = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_sp   = nn.LayerNorm(dim)
        self.norm_sP   = nn.LayerNorm(dim)
        self.ffn_sp    = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(),
                                        nn.Linear(dim*2, dim), nn.Dropout(dropout))
        self.ffn_sP    = nn.Sequential(nn.Linear(dim, dim*2), nn.GELU(),
                                        nn.Linear(dim*2, dim), nn.Dropout(dropout))
        self.cross_pp = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.cross_pP = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_cp  = nn.LayerNorm(dim)
        self.norm_cP  = nn.LayerNorm(dim)
        self.gate_p = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.gate_P = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.pool_p = nn.Linear(dim, 1)
        self.pool_P = nn.Linear(dim, 1)
    def forward(self, patch_seq, pixel_seq):
        B, T, D = patch_seq.shape
        p = patch_seq + self.pos_emb
        P = pixel_seq + self.pos_emb
        p_self, _ = self.self_attn_p(p, p, p)
        p_self = self.norm_sp(p + p_self)
        p_self = self.norm_sp(p_self + self.ffn_sp(p_self))
        P_self, _ = self.self_attn_P(P, P, P)
        P_self = self.norm_sP(P + P_self)
        P_self = self.norm_sP(P_self + self.ffn_sP(P_self))
        p2P, _ = self.cross_pp(p_self, P_self, P_self)
        P2p, _ = self.cross_pP(P_self, p_self, p_self)
        cp = self.norm_cp(torch.cat([p_self, p2P], dim=-1))
        g_p = self.gate_p(cp)
        p_fused = g_p * p2P + (1-g_p) * p_self
        cP = self.norm_cP(torch.cat([P_self, P2p], dim=-1))
        g_P = self.gate_P(cP)
        P_fused = g_P * P2p + (1-g_P) * P_self
        w_p = F.softmax(self.pool_p(p_fused), dim=1)
        w_P = F.softmax(self.pool_P(P_fused), dim=1)
        p_final = (p_fused * w_p).sum(dim=1)
        P_final = (P_fused * w_P).sum(dim=1)
        return torch.cat([p_final, P_final], dim=1)

# —— 主网络 TS2GNet —— #
class TS2GNet(nn.Module):
    def __init__(
        self,
        classes,
        HSI_Data_Shape_H,
        HSI_Data_Shape_W,
        HSI_Data_Shape_C,
        patch_size,
        num_timesteps=3,
        dropout=0.1
    ):
        super().__init__()
        self.name = 'TS2GNet'
        self.classes = classes
        self.HSI_Data_Shape_H = HSI_Data_Shape_H
        self.HSI_Data_Shape_W = HSI_Data_Shape_W
        self.band = HSI_Data_Shape_C
        self.patch_size = patch_size
        self.num_timesteps = num_timesteps

        self.gma = MS2F(channelNumber=self.band, patch_size=self.patch_size)
        self.se_block_patch = SE(self.band)

        # Pixel 支路：ACB + Transformer (LSRC)
        self.acb_pixel = ACB(self.band)
        self.pixel_transformer = LRTM(
            dim=self.band, depth=2, heads=4, dim_head=16, mlp_dim=128, dropout=dropout
        )
        self.fc_pixel = nn.Linear(self.band, 256)

        # Patch 支路卷积层
        self.conv31 = nn.Conv2d(32, 32, 7, padding=3, bias=False)
        self.bn31   = nn.BatchNorm2d(32)
        self.drop31 = nn.Dropout2d(dropout)
        self.conv32 = nn.Conv2d(32, 64, 7, padding=3, bias=False)
        self.bn32   = nn.BatchNorm2d(64)
        self.drop32 = nn.Dropout2d(dropout)
        self.conv33 = nn.Conv2d(64, 128, 5, padding=2, bias=False)
        self.bn33   = nn.BatchNorm2d(128)
        self.drop33 = nn.Dropout2d(dropout)
        self.conv34 = nn.Conv2d(128,256, 5, padding=2, bias=False)
        self.bn34   = nn.BatchNorm2d(256)
        self.drop34 = nn.Dropout2d(dropout)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        if self.num_timesteps > 1:
            self.multi_fusion = TS2FM(
                dim=256, heads=4, dropout=dropout, num_frames=self.num_timesteps
            )
            self.final_classification = nn.Linear(512, self.classes)
            self.finally_fc_classification = self.final_classification
        else:
            self.finally_fc_classification = nn.Linear(256*2, self.classes)

        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patchX, pixelX):
        if patchX.dim() == 4:
            patchX = patchX.unsqueeze(1)
            pixelX = pixelX.unsqueeze(1)

        B, T, C, H, W = patchX.shape
        patch_feats, pixel_feats = [], []

        for t in range(T):
            # Pixel 支路
            pix = pixelX[:, t]
            pix = pix.unsqueeze(-1).unsqueeze(-1)
            pix = self.acb_pixel(pix).squeeze(-1).squeeze(-1)
            pix = pix.unsqueeze(1)
            pix = self.pixel_transformer(pix)[:, 0]
            pix = self.relu(self.fc_pixel(pix))
            pix = self.dropout(pix)
            pixel_feats.append(pix)

            # Patch 支路
            p = patchX[:, t]
            p = self.se_block_patch(p)
            p = self.gma(p)
            x = self.conv31(p)
            x = self.bn31(x); x = self.relu(x); x = self.drop31(x)
            x = F.max_pool2d(x, 2)
            x = self.conv32(x)
            x = self.bn32(x); x = self.relu(x); x = self.drop32(x)
            x = F.max_pool2d(x, 2)
            x = self.conv33(x)
            x = self.bn33(x); x = self.relu(x); x = self.drop33(x)
            x = F.max_pool2d(x, 2)
            x = self.conv34(x)
            x = self.bn34(x); x = self.relu(x); x = self.drop34(x)
            x = self.global_pooling(x).view(B, -1)
            patch_feats.append(x)

        patch_seq = torch.stack(patch_feats, dim=1)
        pixel_seq = torch.stack(pixel_feats, dim=1)

        if T > 1:
            fused = self.multi_fusion(patch_seq, pixel_seq)
            logits = self.final_classification(fused)
        else:
            single = torch.cat([patch_seq[:,0], pixel_seq[:,0]], dim=1)
            logits = self.finally_fc_classification(single)

        out = F.softmax(logits, dim=1)
        return out, out
