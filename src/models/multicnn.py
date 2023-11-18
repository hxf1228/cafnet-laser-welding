import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 use_conv1=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_conv1 = use_conv1
        if use_conv1:
            self.fc1 = nn.Conv1d(in_features, hidden_features, kernel_size=3, stride=1, padding='same')
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        if use_conv1:
            self.fc2 = nn.Conv1d(hidden_features, out_features, kernel_size=3, stride=1, padding='same')
        else:
            self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.use_conv1:
            x = x.transpose(1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        if self.use_conv1:
            x = x.transpose(1, 2)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(nn.Module):
    def __init__(self, in_dim_k, in_dim_q, out_dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = out_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(in_dim_q, out_dim, bias=qkv_bias)
        self.kv = nn.Linear(in_dim_k, out_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkmatrix = None

    def forward(self, x, x_q):
        B, Nk, Ck = x.shape
        B, Nq, Cq = x_q.shape
        q = self.q(x_q).reshape(B, Nq, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        kv = self.kv(x).reshape(B, Nk, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = q.squeeze(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        self.qkmatrix = attn
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, self.qkmatrix


class AttentionBlock(nn.Module):

    def __init__(self, in_dim_k, in_dim_q, out_dim, num_heads, mlp_ratio=2., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_conv1=False):
        super().__init__()
        self.norm1_q = norm_layer(in_dim_q)
        self.norm1_k = norm_layer(in_dim_k)
        self.attn = Attention(in_dim_k=in_dim_k, in_dim_q=in_dim_q,
                              out_dim=out_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(out_dim)
        mlp_hidden_dim = int(out_dim * mlp_ratio)
        self.mlp = Mlp(in_features=out_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,
                       drop=drop, use_conv1=use_conv1)

    def forward(self, xk, xq):
        x, a = self.attn(self.norm1_k(xk), self.norm1_q(xq))
        x = self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=3,
                 drop=0., act_layer=nn.SiLU, norm_layer=nn.BatchNorm1d):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, 1, bias=False),
            norm_layer(hidden_dim),
            act_layer(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=pad, groups=hidden_dim, bias=False),
            norm_layer(hidden_dim),
            act_layer(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(hidden_dim, out_dim, 1, bias=False),
            norm_layer(out_dim)
        )
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        return


def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding="same"):
    return nn.Sequential(nn.Conv1d(in_channels, out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding),
                         nn.BatchNorm1d(out_channels),
                         nn.LeakyReLU(inplace=True),
                         nn.MaxPool1d(2, 1))


class SignalEncoder(nn.Module):
    def __init__(self):
        super(SignalEncoder, self).__init__()

        self.input_dim = 64
        self.conv1 = conv1d_block(1, 32)
        self.conv2 = conv1d_block(32, 64)
        self.conv3 = conv1d_block(64, 128)
        self.conv4 = conv1d_block(128, self.input_dim)
        # self.conv1 = InvertedResidual(in_dim=1, out_dim=32)
        # self.conv2 = InvertedResidual(in_dim=32, out_dim=64)
        # self.conv3 = InvertedResidual(in_dim=64, out_dim=128)
        # self.conv4 = InvertedResidual(128, self.input_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class SignalEncoderSimple(nn.Module):
    def __init__(self):
        super(SignalEncoderSimple, self).__init__()

        self.input_dim = 64
        self.conv1 = conv1d_block(1, 32)
        self.conv2 = conv1d_block(32, 64)
        self.conv3 = conv1d_block(64, 128)
        self.conv4 = conv1d_block(128, self.input_dim)
        # self.conv1 = InvertedResidual(in_dim=1, hidden_dim=1, out_dim=32)
        # self.conv2 = InvertedResidual(in_dim=32, hidden_dim=32, out_dim=64)
        # self.conv3 = InvertedResidual(in_dim=64, hidden_dim=64, out_dim=128)
        # self.conv4 = InvertedResidual(in_dim=128, hidden_dim=128, out_dim=self.input_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class SignalEncoderWide(nn.Module):
    def __init__(self):
        super(SignalEncoderWide, self).__init__()

        self.input_dim = 64
        self.conv1 = conv1d_block(1, 16)
        self.conv2 = conv1d_block(16, 128)
        # self.conv3 = conv1d_block(64, 128)
        self.conv4 = conv1d_block(128, self.input_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        x = self.conv4(x)
        return x


# class MatrixCNN(nn.Module):
#     def __init__(self):
#         super(MatrixCNN, self).__init__()
#         self.acoustic_model = SignalEncoderSimple()
#         self.photodiode_model = SignalEncoderSimple()
#         self.input_dim = self.acoustic_model.input_dim
#         self.fc_dim = self.input_dim * 2  # self.input_dim * 2
#
#         self.classifier_1 = nn.Sequential(nn.Linear(self.fc_dim, 3), )
#
#         self.last_feature = None
#         self.conv2_feature = None
#
#     def forward(self, x_acoustic, x_photodiode):
#         acoustic = self.acoustic_model.conv1(x_acoustic)
#         photodiode = self.photodiode_model.conv1(x_photodiode)
#         acoustic = self.acoustic_model.conv2(acoustic)
#         photodiode = self.photodiode_model.conv2(photodiode)
#         acoustic = self.acoustic_model.conv3(acoustic)
#         photodiode = self.photodiode_model.conv3(photodiode)
#         acoustic = self.acoustic_model.conv4(acoustic)
#         photodiode = self.photodiode_model.conv4(photodiode)
#
#         acoustic_pooled = acoustic.mean([-1])
#         photodiode_pooled = photodiode.mean([-1])
#         x = torch.cat((acoustic_pooled, photodiode_pooled), dim=-1)
#         self.last_feature = x
#         x1 = self.classifier_1(x)
#         return x1


class MultiCNN(nn.Module):
    def __init__(self):
        super(MultiCNN, self).__init__()
        self.acoustic_model = SignalEncoderSimple()
        self.photodiode_model = SignalEncoderSimple()
        self.input_dim = self.acoustic_model.input_dim
        self.fc_dim = self.input_dim * 2  # self.input_dim * 2

        self.classifier_1 = nn.Sequential(nn.Linear(self.fc_dim, 3), )

        self.last_feature = None
        self.conv2_feature = None

    def forward(self, x_acoustic, x_photodiode):
        acoustic = self.acoustic_model.conv1(x_acoustic)
        photodiode = self.photodiode_model.conv1(x_photodiode)
        acoustic = self.acoustic_model.conv2(acoustic)
        photodiode = self.photodiode_model.conv2(photodiode)
        acoustic = self.acoustic_model.conv3(acoustic)
        photodiode = self.photodiode_model.conv3(photodiode)
        acoustic = self.acoustic_model.conv4(acoustic)
        photodiode = self.photodiode_model.conv4(photodiode)

        acoustic_pooled = acoustic.mean([-1])
        photodiode_pooled = photodiode.mean([-1])
        x = torch.cat((acoustic_pooled, photodiode_pooled), dim=-1)
        self.last_feature = x
        x1 = self.classifier_1(x)
        return x1


class EAMultiCNN(nn.Module):
    def __init__(self):
        super(EAMultiCNN, self).__init__()
        self.acoustic_model = SignalEncoderSimple()
        self.photodiode_model = SignalEncoderSimple()
        self.input_dim = self.acoustic_model.input_dim
        self.fc_dim = self.input_dim * 2  # self.input_dim * 2

        self.ap1 = AttentionBlock(in_dim_k=self.input_dim, in_dim_q=self.input_dim,
                                  out_dim=self.input_dim, num_heads=1)
        self.pa1 = AttentionBlock(in_dim_k=self.input_dim, in_dim_q=self.input_dim,
                                  out_dim=self.input_dim, num_heads=1)

        self.classifier_1 = nn.Sequential(nn.Linear(self.fc_dim, 3), )

        self.last_feature = None
        self.conv2_feature = None
        self.att_feature = None

    def forward(self, x_acoustic, x_photodiode):
        acoustic = self.acoustic_model.conv1(x_acoustic)
        photodiode = self.photodiode_model.conv1(x_photodiode)
        acoustic = self.acoustic_model.conv2(acoustic)
        photodiode = self.photodiode_model.conv2(photodiode)
        self.conv2_feature = torch.cat((acoustic, photodiode), dim=-1)

        proj_x_a = acoustic.permute(0, 2, 1)
        proj_x_p = photodiode.permute(0, 2, 1)

        h_av = self.ap1(proj_x_p, proj_x_a)
        h_va = self.pa1(proj_x_a, proj_x_p)

        h_av = h_av.permute(0, 2, 1)
        h_va = h_va.permute(0, 2, 1)

        acoustic = h_av + acoustic
        photodiode = h_va + photodiode
        self.att_feature = torch.cat((acoustic, photodiode), dim=-1)

        acoustic = self.acoustic_model.conv3(acoustic)
        photodiode = self.photodiode_model.conv3(photodiode)


        acoustic = self.acoustic_model.conv4(acoustic)
        photodiode = self.photodiode_model.conv4(photodiode)

        acoustic_pooled = acoustic.mean([-1])
        photodiode_pooled = photodiode.mean([-1])
        x = torch.cat((acoustic_pooled, photodiode_pooled), dim=-1)
        self.last_feature = x
        x1 = self.classifier_1(x)
        #self.last_feature = x1
        return x1
