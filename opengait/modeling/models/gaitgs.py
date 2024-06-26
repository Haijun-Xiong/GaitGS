from ..modules import SeparateFCs, BasicConv3d
from ..base_model import BaseModel
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
from utils import clones
import numpy as np

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        ret = self.conv(x)
        return ret
    
class B3D(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False, **kwargs):
        super(B3D, self).__init__()
        self.conv3d_1 = BasicConv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=bias, **kwargs)
        self.conv3d_2 = BasicConv3d(in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=bias, **kwargs)
        self.conv3d_3 = BasicConv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=bias, **kwargs)

    def forward(self, x):
        x = F.leaky_relu(self.conv3d_1(x) + self.conv3d_2(x) + self.conv3d_3(x))
        return x
    
class STEM(nn.Module):
    def __init__(self, in_channels, out_channels, halving, bias=False, **kwargs):
        super(STEM, self).__init__()
        self.halving = halving
        self.plain = B3D(in_channels, out_channels, bias, **kwargs)
        self.shortcut = B3D(in_channels, out_channels, bias, **kwargs)

    def forward(self, x):
        plain_feat = self.plain(x)
        h = x.size(3)
        split_size = int(h // 2**self.halving)
        shortcut_feat = x.split(split_size, 3)
        shortcut_feat = torch.cat([self.shortcut(_) for _ in shortcut_feat], 3)
        feat = plain_feat + shortcut_feat
        return feat


class MCMs(nn.Module):
    def __init__(self, in_channels, squeeze=4, parts_num=32):
        super(MCMs, self).__init__()
        hidden_dim = int(in_channels // squeeze)
        self.parts_num = parts_num
        # MTB1
        conv3x1 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 1))
        self.conv1d3x1 = clones(conv3x1, parts_num)
        self.avg_pool3x1 = nn.AvgPool1d(3, stride=1, padding=1)
        self.max_pool3x1 = nn.MaxPool1d(3, stride=1, padding=1)
        # MTB2
        conv3x3 = nn.Sequential(
            BasicConv1d(in_channels, hidden_dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            BasicConv1d(hidden_dim, in_channels, 3, padding=1))
        self.conv1d3x3 = clones(conv3x3, parts_num)
        self.avg_pool3x3 = nn.AvgPool1d(5, stride=1, padding=2)
        self.max_pool3x3 = nn.MaxPool1d(5, stride=1, padding=2)
        # Temporal Pooling, TP
        self.TP = torch.max

    def forward(self, x):
        n, c, s, p = x.size()
        x = x.permute(3, 0, 1, 2).contiguous()  # [p, n, c, s]
        feature = x.split(1, 0)  # [[1, n, c, s], ...]
        x = x.view(-1, c, s)
        # MTB1: ConvNet1d & Sigmoid
        logits3x1 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x1, feature)], 0)
        scores3x1 = torch.sigmoid(logits3x1)
        # MTB1: Template Function
        feature3x1 = self.avg_pool3x1(x) + self.max_pool3x1(x)
        feature3x1 = feature3x1.view(p, n, c, s)
        feature3x1 = feature3x1 * scores3x1
        # MTB2: ConvNet1d & Sigmoid
        logits3x3 = torch.cat([conv(_.squeeze(0)).unsqueeze(0)
                               for conv, _ in zip(self.conv1d3x3, feature)], 0)
        scores3x3 = torch.sigmoid(logits3x3)
        # MTB2: Template Function
        feature3x3 = self.avg_pool3x3(x) + self.max_pool3x3(x)
        feature3x3 = feature3x3.view(p, n, c, s)
        feature3x3 = feature3x3 * scores3x3  # [p, n, c, s]
        # Temporal Pooling
        outs = feature3x1 + feature3x3
        outs = rearrange(outs, 'p n c s -> n c s p')
        return outs
    
    
class CAPE(nn.Module):
    def __init__(self, dim, kernel_size):
        super(CAPE, self).__init__()
        padding = kernel_size // 2
        self.conv1d = BasicConv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, stride=1, padding=padding, groups=dim)
        
    def forward(self, x):
        n = x.size(0)
        x = rearrange(x, 'n c s p -> (n p) c s')
        x = self.conv1d(x)
        x = rearrange(x, '(n p) c s -> n c s p', n=n)
        return x


class L_Transformer(nn.Module):
    def __init__(self, dim, depth, out_dim, parts_num, num_head):
        super(L_Transformer, self).__init__()

        attn = nn.TransformerEncoderLayer(dim, num_head, 256, norm_first=True, batch_first=True, activation='gelu', dropout=0.2)
        self.attn = nn.TransformerEncoder(attn, depth)
        self.separate_mlp = SeparateFCs(parts_num, dim, out_dim)
        
    def forward(self, x):
        n= x.size(0)
        x = rearrange(x, 'n c s p -> (n p) s c')
        x = self.attn(x)
        x = rearrange(x[:, 0], '(n p) c -> n c p', n=n)
        x = self.separate_mlp(x)
        return x
    

class MGFE(nn.Module):
    def __init__(self, in_c):
        super(MGFE, self).__init__()

        self.conv3d = nn.Sequential(
            BasicConv3d(1, in_c[0], kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)), 
            nn.LeakyReLU(inplace=True),
            BasicConv3d(in_c[0], in_c[0], kernel_size=1, stride=1, padding=0), 
            nn.LeakyReLU(inplace=True)
        )
        self.STEM_S0 = STEM(in_c[0], in_c[1], halving=3)
        self.STEM_S1 = nn.Sequential(
            STEM(in_c[1], in_c[2], halving=3),
            STEM(in_c[2], in_c[2], halving=3)
        )

        self.STEM_C0 = STEM(in_c[0], in_c[1], halving=3)
        self.STEM_C1 = nn.Sequential(
            STEM(in_c[1], in_c[2], halving=3),
            STEM(in_c[2], in_c[2], halving=3)
        )
        
        self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.UTA0 = BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0))
        self.UTA1 = BasicConv3d(in_c[1], in_c[1], kernel_size=(3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0))
        self.UTA2 = BasicConv3d(in_c[2], in_c[2], kernel_size=(3, 1, 1), stride=(3, 1, 1), padding=(0, 0, 0))
        
    def forward(self, x):
        x = self.conv3d(x)
        x1 = self.UTA0(x)

        x = self.STEM_S0(x)
        x = self.MaxPool0(x)
        x1 = self.STEM_C0(x1)
        x1 = self.MaxPool0(x1)
        x1 = x1 + self.UTA1(x)

        x = self.STEM_S1(x)
        x1 = self.STEM_C1(x1)
        x1 = x1 + self.UTA2(x)
        return x, x1

class PIEG(nn.Module):
    def __init__(self, view_num, in_channel, eps = 1e-6):
        super(PIEG, self).__init__()
        self.eps = eps
        self.P = nn.Parameter(torch.tensor(6.5), requires_grad=True)
        self.view_cls = nn.Linear(in_channel, view_num, bias=False)
        self.V_view = nn.Parameter(torch.randn(view_num, int(in_channel/2)))
    
    def TP(self, x):
        return torch.max(x, 2)[0]

    def gem(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.P), (x.size(-2), x.size(-1))).pow(1. / self.P)
    
    def forward(self, x, x_c):
        x = self.TP(x)
        x_c = self.TP(x_c)
        x = torch.cat((x, x_c), dim=1)
        x = self.gem(x)
        x = x.squeeze(-1).squeeze(-1)

        view_p_predict = self.view_cls(x)
        view_y_predict = torch.max(view_p_predict, dim=-1)[1]
        view_prior = self.V_view[view_y_predict]
        
        prior = view_prior
        
        return prior, view_p_predict


class CATM(nn.Module):
    def __init__(self, dim, depth, kernel_size, out_dim, parts_num, num_head):
        super(CATM, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, dim, 1, parts_num))
        self.CAPE = CAPE(dim=dim, kernel_size=kernel_size)
        self.L_Transformer = L_Transformer(dim=dim, depth=depth, out_dim=out_dim, parts_num=parts_num, num_head=num_head)
        
    def forward(self, x, prior):
        n, c, s, p = x.size()
        x = x + self.CAPE(x)
        cls_token = repeat(self.cls_token, '1 c 1 p -> n c 1 p', n=n)
        x = torch.cat((cls_token, x), dim=2)
        prior = repeat(prior, 'n c -> n c s p', s=s+1, p=p)
        x = x + prior
        outs = self.L_Transformer(x)
        return outs

        
class GaitGS(BaseModel):
    def __init__(self, *args, **kargs):
        super(GaitGS, self).__init__(*args, **kargs)

    def build_network(self, model_cfg):
        in_c = model_cfg['channels']
        view_num = model_cfg['view_num']
        parts_num = model_cfg['parts_num']
        out_dim = model_cfg['out_dim']
        depth = model_cfg['depth']
        kernel_size = model_cfg['kernel_size']
        num_head = model_cfg['num_head']
        self.parts_num = parts_num

        self.mgfe = MGFE(in_c)
        self.MCMs_S = MCMs(in_channels=in_c[-1], parts_num=parts_num)
        self.MCMs_C = MCMs(in_channels=in_c[-1], parts_num=parts_num)
        self.PIEG = PIEG(view_num, in_c[-1]*2)
        self.catm_f = CATM(in_c[-1], depth, kernel_size, out_dim, parts_num, num_head)
        self.catm_c = CATM(in_c[-1], depth, kernel_size, out_dim, parts_num, num_head)
        self.fc = SeparateFCs(parts_num, in_c[-1], out_dim)
        self.fc_L = SeparateFCs(parts_num, in_c[-1], out_dim)
    
    def HP(self, x):
        x = rearrange(x, 'n c s (p h) w -> n c s p (h w)', p=self.parts_num)
        x = x.mean(-1) + x.max(-1)[0]
        return x
    
    def forward(self, inputs):
        ipts, labs, cond, view, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        sils = ipts[0].unsqueeze(1)
        del ipts
        n, _, s, h, w = sils.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            sils = sils.repeat(1, 1, repeat, 1, 1)
        x = sils
        # Convert view&cond into int
        view = [int(int(i)/18) for i in view] 
        view = torch.tensor(view).long().cuda()

        S_f, S_c = self.mgfe(x)

        E_prior, prob_view = self.PIEG(S_f, S_c)
        S_f = self.HP(S_f)
        S_c = self.HP(S_c)
        E_f = self.MCMs_S(S_f)
        S_fl = torch.max(E_f, dim=2)[0]  # [n, p, c]
        E_c = self.MCMs_C(S_c)
        S_cl = torch.max(E_c, dim=2)[0]  # [n, p, c]

        S_fg = self.catm_f(E_f, E_prior)
        gait_f = S_fg + self.fc(S_fl)
        S_cg = self.catm_c(E_c, E_prior)
        gait_c = S_cg + self.fc_L(S_cl)

        gait = torch.cat((gait_f, gait_c), dim=2)
        
        embed = gait
        n, _, s, h, w = sils.size()
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'view_softmax':{'logits': prob_view.unsqueeze(-1), 'labels': view}
            },
            'visual_summary': {
                'image/sils': sils.view(n*s, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval