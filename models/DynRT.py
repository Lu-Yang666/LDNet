import copy

import math
import timm
import torch
import torch.nn as nn
from transformers import AutoModel, BertModel, AutoConfig,ViTModel
import torch.nn.functional as F
from utils import decode_nnw_nsw_thw_mat, decode_nnw_thw_mat, decode_pointer_mat
import numpy as np

class cls_layer_both(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cls_layer_both, self).__init__()
        self.proj_norm = LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, lang_feat, img_feat):
        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = self.proj(proj_feat)

        return proj_feat

class SoftRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel,
                 pooling='attention', reduction=2):
        super(SoftRoutingBlock, self).__init__()
        self.pooling = pooling
        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, x, tau, masks):
        if self.pooling == 'avg':
            x = x.transpose(1, 2)
            x = self.pool(x)
            logits = self.mlp(x.squeeze(-1))
        elif self.pooling == 'fc':
            b, _, c = x.size()
            mask = self.make_mask(x).squeeze(1).squeeze(1).unsqueeze(2)  # (8, 1, 1, 49) -> (8, 49, 1)
            scores = self.pool(x)  # (8, 49, 1)
            scores = scores.masked_fill(mask, -1e9)
            scores = F.softmax(scores, dim=1)
            _x = x.mul(scores)
            x = torch.sum(_x, dim=1)
            logits = self.mlp(x)
        alpha = F.softmax(logits, dim=-1)
        return alpha

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


class HardRoutingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, pooling='attention', reduction=2):
        super(HardRoutingBlock, self).__init__()
        self.pooling = pooling
        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pooling == 'fc':
            self.pool = nn.Linear(in_channel, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, out_channel, bias=True),
        )

    def forward(self, v, tau, masks):
        if self.pooling == 'avg':
            v = v.transpose(1, 2)
            v = self.pool(v)
            logits = self.mlp(v.squeeze(-1))
        elif self.pooling == 'fc':
            b, _, c = v.size()
            mask = self.make_mask(v).squeeze(1).squeeze(1).unsqueeze(2)
            scores = self.pool(v)  # (8, 49, 1)
            scores = scores.masked_fill(mask, -1e9)
            scores = F.softmax(scores, dim=1)
            _v = v.mul(scores)
            v = torch.sum(_v, dim=1)
            logits = self.mlp(v)
            # print(logits)
        alpha = self.gumbel_softmax(logits, -1, tau)
        return alpha

    def gumbel_softmax(self, logits, dim=-1, temperature=0.1):
        gumbels = -torch.empty_like(logits).exponential_().log()
        logits = (logits.log_softmax(dim=dim) + gumbels) / temperature
        return F.softmax(logits, dim=dim)

    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)


class mean_Block(nn.Module):
    """
    Self-Attention Routing Block
    """

    def __init__(self, hidden_size, orders):
        super(mean_Block, self).__init__()
        self.len = orders
        self.hidden_size = hidden_size

    def forward(self, x, tau, masks):
        alpha = (1 / self.len) * torch.ones(x.shape[0], self.len).to(x.device)  # (bs, 4)
        return alpha


class SARoutingBlock(nn.Module):
    """
    Self-Attention Routing Block
    """

    def __init__(self, modelopt):
        super(SARoutingBlock, self).__init__()
        self.modelopt = modelopt
        self.linear_v = nn.Linear(modelopt["hidden_size"], modelopt["hidden_size"])  # 768
        self.linear_k = nn.Linear(modelopt["hidden_size"], modelopt["hidden_size"])
        self.linear_q = nn.Linear(modelopt["hidden_size"], modelopt["hidden_size"])
        self.linear_merge = nn.Linear(modelopt["hidden_size"], modelopt["hidden_size"])
        if modelopt["routing"] == 'hard':
            self.routing_block = HardRoutingBlock(modelopt["hidden_size"], modelopt["orders"], modelopt["pooling"])
        elif modelopt["routing"] == 'soft':
            self.routing_block = SoftRoutingBlock(modelopt["hidden_size"], modelopt["orders"], modelopt["pooling"])
        elif modelopt["routing"] == 'mean':
            self.routing_block = mean_Block(modelopt["hidden_size"], modelopt["orders"])
        self.dropout = nn.Dropout(modelopt["dropout"])

    def forward(self, v, k, q, masks, tau, training):
        n_batches = q.size(0)
        alphas = self.routing_block(v, tau, masks)  # (bs, 4)
        if self.modelopt["BINARIZE"]:
            if not training:
                alphas = self.argmax_binarize(alphas)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.modelopt["multihead"],  # 2
            int(self.modelopt["hidden_size"] / self.modelopt["multihead"])
        ).transpose(1, 2)  # (bs,2, , 384)
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.modelopt["multihead"],
            int(self.modelopt["hidden_size"] / self.modelopt["multihead"])
        ).transpose(1, 2)  # (bs,2, , 384)
        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.modelopt["multihead"],
            int(self.modelopt["hidden_size"] / self.modelopt["multihead"])
        ).transpose(1, 2)  # (bs,2, , 384)
        att_list = self.routing_att(v, k, q, masks)  # (bs, order_num, head_num, grid_num, grid_num) (bs, 4, 4, 49, 49)
        att_map = torch.einsum('bl,blcnm->bcnm', alphas, att_list)  # (bs, 4), (bs, 4, 4, 49, 49) - > (bs, 4, 49, 49)
        atted = torch.matmul(att_map,
                             v)  # (bs, 4, 49, [49]) * (bs, 4, [49],192) - > (bs, 4, 49, 192) mul [49, 49]*[49, 192],
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.modelopt["hidden_size"]
        )  # (bs, 49, 768)
        atted = self.linear_merge(atted)  # (bs, 4, 768)
        return atted

    def routing_att(self, value, key, query, masks):
        d_k = query.size(-1)  # masks [[bs, 1, 1, 49], [bs, 1, 49, 49], [bs, 1, 49, 49], [bs, 1, 49, 49]]
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (32,2,100,49)
        # k q v [32, 2, 49, 384] key (32, 2, 100, 384) query [32, 2, 49, 384]
        for i in range(len(masks)):
            mask = masks[i]  # (bs, 1, 100, 49) (32,1,1,49)
            scores_temp = scores.masked_fill(mask, -1e9)
            att_map = F.softmax(scores_temp, dim=-1)
            att_map = self.dropout(att_map)
            if i == 0:
                att_list = att_map.unsqueeze(1)  # (bs, 1, 4, 49, 49)
            else:
                att_list = torch.cat((att_list, att_map.unsqueeze(1)), 1)  # (bs, 2, 4, 49, 49) -> (bs, 3, 4, 49, 49)
        return att_list

    def argmax_binarize(self, alphas):
        n = alphas.size()[0]
        out = torch.zeros_like(alphas)
        indexes = alphas.argmax(-1)
        out[torch.arange(n), indexes] = 1
        return out


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------
class FC(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0, activation=None):
        super(FC, self).__init__()
        self.hasactivation = activation is not None
        self.linear = nn.Linear(input_dim, output_dim)
        if activation is not None:
            self.activation = nn.ReLU(inplace=True)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.linear(x)
        if self.hasactivation:
            x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0, activation=None):
        super(MLP, self).__init__()
        self.fc = FC(input_dim, hidden_dim, dropout=dropout, activation=activation)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.linear(self.fc(x))


class FFN(nn.Module):
    def __init__(self, opt):
        super(FFN, self).__init__()

        self.mlp = MLP(
            input_dim=opt["hidden_size"],
            hidden_dim=opt["ffn_size"],
            output_dim=opt["hidden_size"],
            dropout=opt["dropout"],
            activation="ReLU"
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------
class MHAtt(nn.Module):
    def __init__(self, opt):
        super(MHAtt, self).__init__()
        self.opt = opt
        self.linear_v = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_k = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_q = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.linear_merge = nn.Linear(opt["hidden_size"], opt["hidden_size"])
        self.dropout = nn.Dropout(opt["dropout"])

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)
        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)
        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)
        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.opt["multihead"],
            int(self.opt["hidden_size"] / self.opt["multihead"])
        ).transpose(1, 2)
        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.opt["hidden_size"]
        )
        atted = self.linear_merge(atted)
        return atted

    def att(self, value, key, query, mask):
        # print(query.shape)
        # print(key.shape)
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        if mask is not None:
            # print(scores.shape)
            scores = scores.masked_fill(mask, -1e9)
        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)
        return torch.matmul(att_map, value)


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class multiTRAR_SA_block(nn.Module):
    def __init__(self, modelopt):
        super(multiTRAR_SA_block, self).__init__()
        self.mhatt1 = SARoutingBlock(modelopt)
        self.mhatt2 = MHAtt(modelopt)
        self.ffn = FFN(modelopt)
        self.dropout1 = nn.Dropout(modelopt["dropout"])
        self.norm1 = LayerNorm(modelopt["hidden_size"])
        self.dropout2 = nn.Dropout(modelopt["dropout"])
        self.norm2 = LayerNorm(modelopt["hidden_size"])
        self.dropout3 = nn.Dropout(modelopt["dropout"])
        self.norm3 = LayerNorm(modelopt["hidden_size"])

    def forward(self, lang_feat, img_feat, img_feat_mask, lang_feat_mask, tau, training):  # x (64, 49, 512) y

        lang_feat = self.norm1(lang_feat + self.dropout1(
            self.mhatt1(v=img_feat, k=img_feat, q=lang_feat, masks=img_feat_mask, tau=tau, training=training)
        ))  # (64, 49, 512) # (bs, 49, 768)
        lang_feat = self.norm2(lang_feat + self.dropout2(
            self.mhatt2(v=lang_feat, k=lang_feat, q=lang_feat, mask=lang_feat_mask)
        ))
        lang_feat = self.norm3(lang_feat + self.dropout3(
            self.ffn(lang_feat)
        ))
        return lang_feat


def getImgMasks(scale=16, order=2):
    """
    :param scale: Feature Map Scale
    :param order: Local Window Size, e.g., order=2 equals to windows size (5, 5)
    :return: masks = (scale**2, scale**2)
    """
    masks = []
    _scale = scale
    assert order < _scale, 'order size be smaller than feature map scale'

    for i in range(_scale):
        for j in range(_scale):
            mask = np.ones([_scale, _scale], dtype=np.float32)
            for x in range(i - order, i + order + 1, 1):
                for y in range(j - order, j + order + 1, 1):
                    if (0 <= x < _scale) and (0 <= y < _scale):
                        mask[x][y] = 0
            mask = np.reshape(mask, [_scale * _scale])
            masks.append(mask)
            # print(mask)
    masks = np.array(masks)
    masks = np.asarray(masks, dtype=np.bool_)  # 0, 1 -> False True (True mask)
    return masks


# getImgMasks(3,1)
def getMasks_img_multimodal(x_mask, __C):
    mask_list = []  # x_mask [64, 1, 1, 49]
    ORDERS = __C["ORDERS"]
    for order in ORDERS:
        if order == 0:
            mask_list.append(x_mask)
        else:
            mask_img = torch.from_numpy(getImgMasks(__C["IMG_SCALE"], order)).byte().to(x_mask.device)  # (49, 49)
            mask = torch.concat([mask_img] * (__C["len"] // (__C["IMG_SCALE"] * __C["IMG_SCALE"])), dim=0)
            mask = torch.concat([mask, mask_img[:(__C["len"] % (__C["IMG_SCALE"] * __C["IMG_SCALE"])), :]])
            mask = torch.logical_or(x_mask, mask)  # (64, 1, max_len, grid_num)
            mask_list.append(mask)
    return mask_list


class DynRT_ED(nn.Module):
    def __init__(self, modelopt):
        super(DynRT_ED, self).__init__()
        self.modelopt = modelopt
        self.tau = modelopt["tau_max"]  # 10
        opt_list = []
        for i in range(modelopt["layer"]):  # 4
            opt_copy = copy.deepcopy(modelopt)
            opt_copy["ORDERS"] = modelopt["ORDERS"][:len(modelopt["ORDERS"]) - i]
            opt_copy["orders"] = len(modelopt["ORDERS"]) - i
            opt_list.append(copy.deepcopy(opt_copy))
        self.dec_list = nn.ModuleList([multiTRAR_SA_block(opt_list[-(i + 1)]) for i in range(modelopt["layer"])])

    # TRAR
    def forward(self, lang_feat, img_feat, lang_feat_mask, img_feat_mask):  # x_img,y_lang
        # y text (bs, max_len, dim) x img (bs, gird_num, dim) y_mask (bs, 1, 1, max_len) x_mask (bs, 1, 1, grid_num)
        img_feat_mask = getMasks_img_multimodal(img_feat_mask, self.modelopt)
        for i, dec in enumerate(self.dec_list):
            # Input encoder last hidden vector
            # And obtain decoder last hidden vectors
            lang_feat = dec(lang_feat, img_feat, img_feat_mask[:i + 1], lang_feat_mask, self.tau,
                           self.training)  # (4, 360, 768)
        return lang_feat, img_feat

    def set_tau(self, tau):
        self.tau = tau

