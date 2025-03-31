import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


t_stride = 1

model_path = {
    's_sip_nods_s4': '/mnt/WXRC0020/users/junhao.zhang/tmp/slowfast/tools/s_sip_nods.pth',
}
def conv_3xnxn(inp, oup, kernel_size=3, stride=3,padding=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, padding, padding))


def conv_1xnxn(inp, oup, kernel_size=3, stride=3,padding=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, padding, padding))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#todo add
class Involution(nn.Module):
    def __init__(self, kernel_size, in_channel=4, stride=1, group=1, ratio=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.stride = stride
        self.group = group
        assert self.in_channel % group == 0
        self.group_channel = self.in_channel // group
        self.conv1 = nn.Conv2d(
            self.in_channel,
            self.in_channel // ratio,
            kernel_size=1
        )
        self.bn = nn.BatchNorm2d(in_channel // ratio)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            self.in_channel // ratio,
            self.group * self.kernel_size * self.kernel_size,
            kernel_size=1
        )
        self.avgpool = nn.AvgPool2d(stride, stride) if stride > 1 else nn.Identity()
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2))
        self.max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=int(kernel_size / 2))

    def forward(self, inputs):
        B, C, H, W = inputs.shape

        weight = self.conv2(self.relu(self.bn(self.conv1(inputs))))  # (bs,G*K*K,H//stride,W//stride)
        b, c, h, w = weight.shape
        weight = weight.reshape(b, self.group, self.kernel_size * self.kernel_size, h, w).unsqueeze(
            2)  # (bs,G,1,K*K,H//stride,W//stride)

        x_unfold = self.unfold(inputs)

        x_unfold = x_unfold.reshape(B, self.group, C // self.group, self.kernel_size * self.kernel_size,
                                    H // self.stride, W // self.stride)  # (bs,G,C//G,K*K,H//stride,W//stride)

        out = (x_unfold * weight).sum(dim=3)  # (bs,G,G//C,1,H//stride,W//stride)

        out = out.reshape(B, C, H // self.stride, W // self.stride)  # (bs,C,H//stride,W//stride)

        return out
#todo add
class InvoMLP(nn.Module):
    def __init__(self, dim, ikernel, group, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight1 = Mlp(dim, dim // 4, dim * 2)
        self.reweight2 = Mlp(dim, dim // 4, dim * 2)

        self.proj = nn.Linear(dim, dim)
        self.projcnn = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.invo = Involution(kernel_size=ikernel, in_channel=dim, stride=1, group=group)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.i = ikernel
        self.g = group
    def forward(self, x):
        B, H, W, C = x.shape

        hw = self.invo(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        c = self.mlp_c(x)

        a = (hw + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight1(a).reshape(B, C, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = hw * a[0] + c * a[1]

        x = self.projcnn(x.permute(0,3,1,2)).permute(0,2,3,1)

        return x


#todo add
class HireMLP(nn.Module):
    def __init__(self, dim, attn_drop=0., proj_drop=0., pixel=2,
                 step=1, step_pad_mode='c', pixel_pad_mode='c'):
        super().__init__()
        """
        self.pixel: h and w in inner-region rearrangement
        self.step: s in cross-region rearrangement
        """
        self.pixel = pixel
        self.step = step
        self.step_pad_mode = step_pad_mode
        self.pixel_pad_mode = pixel_pad_mode
        print('pixel: {} pad mode: {} step: {} pad mode: {}'.format(
            pixel, pixel_pad_mode, step, step_pad_mode))

        self.mlp_h1 = nn.Conv2d(dim * pixel, dim // 2, 1, bias=False)
        self.mlp_h1_norm = nn.BatchNorm2d(dim // 2)
        self.mlp_h2 = nn.Conv2d(dim // 2, dim * pixel, 1, bias=True)
        self.mlp_w1 = nn.Conv2d(dim * pixel, dim // 2, 1, bias=False)
        self.mlp_w1_norm = nn.BatchNorm2d(dim // 2)
        self.mlp_w2 = nn.Conv2d(dim // 2, dim * pixel, 1, bias=True)
        self.mlp_c = nn.Conv2d(dim, dim, 1, bias=True)

        self.act = nn.ReLU()

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        h: H x W x C -> H/pixel x W x C*pixel
        w: H x W x C -> H x W/pixel x C*pixel
        Setting of F.pad: (left, right, top, bottom)
        """

        B, C, H, W = x.shape

        pad_h, pad_w = (self.pixel - H % self.pixel) % self.pixel, (self.pixel - W % self.pixel) % self.pixel
        h, w = x.clone(), x.clone()

        if self.step:
            if self.step_pad_mode == '0':
                h = F.pad(h, (0, 0, self.step, 0), "constant", 0)
                w = F.pad(w, (self.step, 0, 0, 0), "constant", 0)
                h = torch.narrow(h, 2, 0, H)
                w = torch.narrow(w, 3, 0, W)
            elif self.step_pad_mode == 'c':
                h = torch.roll(h, self.step, -2)
                w = torch.roll(w, self.step, -1)
                # h = F.pad(h, (0, 0, self.step, 0), mode='circular')
                # w = F.pad(w, (self.step, 0, 0, 0), mode='circular')
            else:
                raise NotImplementedError("Invalid pad mode.")

        if self.pixel_pad_mode == '0':
            h = F.pad(h, (0, 0, 0, pad_h), "constant", 0)
            w = F.pad(w, (0, pad_w, 0, 0), "constant", 0)
        elif self.pixel_pad_mode == 'c':
            h = F.pad(h, (0, 0, 0, pad_h), mode='circular')
            w = F.pad(w, (0, pad_w, 0, 0), mode='circular')
        elif self.pixel_pad_mode == 'replicate':
            h = F.pad(h, (0, 0, 0, pad_h), mode='replicate')
            w = F.pad(w, (0, pad_w, 0, 0), mode='replicate')
        else:
            raise NotImplementedError("Invalid pad mode.")

        h = h.reshape(B, C, (H + pad_h) // self.pixel, self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C * self.pixel,
                                                                                                     (
                                                                                                                 H + pad_h) // self.pixel,
                                                                                                     W)
        w = w.reshape(B, C, H, (W + pad_w) // self.pixel, self.pixel).permute(0, 1, 4, 2, 3).reshape(B, C * self.pixel,
                                                                                                     H, (
                                                                                                                 W + pad_w) // self.pixel)

        h = self.mlp_h1(h)
        h = self.mlp_h1_norm(h)
        h = self.act(h)
        h = self.mlp_h2(h)

        w = self.mlp_w1(w)
        w = self.mlp_w1_norm(w)
        w = self.act(w)
        w = self.mlp_w2(w)

        h = h.reshape(B, C, self.pixel, (H + pad_h) // self.pixel, W).permute(0, 1, 3, 2, 4).reshape(B, C, H + pad_h, W)
        w = w.reshape(B, C, self.pixel, H, (W + pad_w) // self.pixel).permute(0, 1, 3, 4, 2).reshape(B, C, H, W + pad_w)

        h = torch.narrow(h, 2, 0, H)
        w = torch.narrow(w, 3, 0, W)

        # cross-region arrangement operation
        if self.step and self.step_pad_mode == 'c':
            h = torch.roll(h, -self.step, -2)
            w = torch.roll(w, -self.step, -1)
            # h = F.pad(h, (0, 0, 0, self.step), mode='circular')
            # w = F.pad(w, (0, self.step, 0, 0), mode='circular')
            # h = torch.narrow(h, 2, self.step, H)
            # w = torch.narrow(w, 3, self.step, W)

        c = self.mlp_c(x)

        a = (h + w + c).flatten(2).mean(2).unsqueeze(2).unsqueeze(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(3).unsqueeze(3)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Spa_FC(nn.Module):
    def __init__(self, dim, segment_dim=8,tmp=7, C=3,qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.tmp=tmp
        dim2=C*tmp
        self.mlp_h = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)

        # init weight problem
        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, H, W, C = x.shape

        S = C // self.segment_dim
        tmp=self.tmp
        # H
        h = x.transpose(3,2).reshape(B, T,H*W//tmp,tmp, self.segment_dim, S).permute(0, 1, 2, 4, 3, 5).reshape(B, T,  H*W//tmp,self.segment_dim,tmp* S)
        h = self.mlp_h(h).reshape(B, T,  H*W//tmp,self.segment_dim, tmp,S).permute(0, 1, 2, 4, 3, 5).reshape(B, T, W, H, C).transpose(3,2)
        # W
        w = x.reshape(B, T, H* W//tmp,tmp, self.segment_dim, S).permute(0, 1, 2, 4, 3, 5).reshape(B, T,  H*W//tmp,self.segment_dim, tmp* S)
        w = self.mlp_w(w).reshape(B, T, H*W//tmp,self.segment_dim, tmp,S).permute(0, 1, 2, 4, 3, 5).reshape(B, T, H, W, C)
        # C
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Spe_FC(nn.Module):
    def __init__(self, dim,segment_dim,band,C ,qkv_bias=False, proj_drop=0.):
        super().__init__()

        self.segment_dim =segment_dim
        dim2 = band*C

        self.mlp_t = nn.Linear(dim2, dim2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, H, W, C = x.shape

        S = C // self.segment_dim


        # T
        t = x.reshape(B, T, H, W, self.segment_dim, S).permute(0, 4, 2, 3, 1, 5).reshape(B, self.segment_dim, H, W, T * S)
        t = self.mlp_t(t).reshape(B, self.segment_dim, H, W, T, S).permute(0, 4, 2, 3, 1, 5).reshape(B, T, H, W, C)

        x = t

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class PermutatorBlock(nn.Module):
    def __init__(self, dim, segment_dim,tmp, band,C,mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0):
    # # add todo
    # def __init__(self, dim, segment_dim,tmp, band,C,mlp_ratio=4., qkv_bias=False,drop = 0., attn_drop = 0.,
    #              drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0,pixel = 2, step = 1, step_pad_mode = 'c', pixel_pad_mode = 'c'):

        super().__init__()
        self.norm1 = norm_layer(dim)
        self.s_norm1 = norm_layer(dim)
        self.s_fc = Spe_FC(dim, segment_dim,band,C,qkv_bias=qkv_bias)
        self.fc = Spa_FC(dim, segment_dim=segment_dim, tmp=tmp, C=C,qkv_bias=qkv_bias)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # #add todo
        # self.attn = HireMLP(
        #     dim, attn_drop=attn_drop, pixel=pixel, step=step,
        #     step_pad_mode=step_pad_mode, pixel_pad_mode=pixel_pad_mode)
        # # add todo
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        xs = x + self.s_fc(self.s_norm1(x))
        x = x + self.drop_path(self.fc(self.norm1(xs))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj1 = conv_3xnxn(in_chans, embed_dim//2, kernel_size=1, stride=1,padding=0)
        self.norm1= nn.BatchNorm3d(embed_dim//2)
        self.act=nn.GELU()
        self.proj2 = conv_1xnxn(embed_dim//2, embed_dim, kernel_size=3, stride=1,padding=1)
        self.norm2 = nn.BatchNorm3d(embed_dim)

    def forward(self, x):
        x = self.proj1(x)#3维卷积
        x= self.norm1(x)#正则化
        x=self.act(x)#GELU
        x = self.proj2(x)#1维卷积
        x = self.norm2(x)#正则化
        return x

class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = conv_1xnxn(in_embed_dim, out_embed_dim, kernel_size=3, stride=2,padding=1)
        self.norm=nn.LayerNorm(out_embed_dim)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)  # B, C, T, H, W
        x = x.permute(0, 2, 3, 4, 1)
        x=self.norm(x)
        return x

class SSMLP(nn.Module):
    """ MorphMLP
    """

    def __init__(self,Patch, BAND, CLASSES_NUM,layers,embed_dims,segment_dim):
        super().__init__()
        global t_stride


        num_classes = CLASSES_NUM

        in_chans = 1
        layers = layers
        segment_dim = segment_dim
        mlp_ratios =3
        embed_dims =embed_dims

        tmp = Patch
        qkv_bias = True
        C=int(embed_dims/segment_dim)

        drop_path_rate =0.1
        norm_layer = nn.LayerNorm

        skip_lam = 1.0

        self.num_classes = num_classes

        self.patch_embed1 = PatchEmbed( in_chans=in_chans, embed_dim=embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        # for item in dpr:
        #     print(item)

        # stage1
        self.blocks1 = nn.ModuleList([])
        for i in range(layers):
            self.blocks1.append(
                PermutatorBlock(embed_dims, segment_dim,tmp=tmp,band=BAND,C=C, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, drop_path=dpr[i], skip_lam=skip_lam)
            )

        self.norm = norm_layer(embed_dims)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Classifier head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_fc.mlp_t.weight' in name:
                nn.init.constant_(p, 0)
            if 't_fc.mlp_t.bias' in name:
                nn.init.constant_(p, 0)
            if 't_fc.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_fc.proj.bias' in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pretrained_model(self, cfg):
        if cfg.MORPH.PRETRAIN_PATH:
            checkpoint = torch.load(cfg.MORPH.PRETRAIN_PATH, map_location='cpu')
            if self.num_classes != 1000:
                del checkpoint['head.weight']
                del checkpoint['head.bias']
            return checkpoint
        else:
            return None

    def forward_features(self, x):
        x=x.view(x.shape[0],1,x.shape[1],x.shape[2],x.shape[3])#todo x=[21,200,7,7]->[21,1,200,7,7]
        x = self.patch_embed1(x)#todo [21,1,200,7,7]->[21,64,200,7,7]
        # B,C,T,H,W -> B,T,H,W,C
        x = x.permute(0, 2, 3,4, 1)#todo [21,64,200,7,7]->[21,200,7,7,64]

        for blk in self.blocks1:
            x = blk(x)

        #x = x.permute(0, 4, 2, 3, 1)
        B, T,H, W, C = x.shape
        x = x.reshape(B, -1, C)#todo [21,200,7,7,64]->[21,9800,64]
        return x

    def forward(self, x,bool):

        x = self.forward_features(x)#todo [21,200,7,7]->[21,9800,64]
        #x = self.avgpool(x)
        #x = self.norm(x.squeeze())
        x = self.norm(x)#todo [21,9800,64]->
        #x = torch.flatten(x, 1)

        return x.mean(1),self.head(x.mean(1))#todo .mean(1)计算每一行的平均值，
        #return x, self.head(x)


