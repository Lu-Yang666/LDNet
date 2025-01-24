import copy
import gc

import math
import timm
import torch
import torch.nn as nn
from transformers import AutoModel, BertModel, AutoConfig, ViTModel
import torch.nn.functional as F
from utils import decode_nnw_nsw_thw_mat, decode_nnw_thw_mat, decode_pointer_mat
import numpy as np
from models.DynRT import DynRT_ED, cls_layer_both
from utils import windowed_queue_iter


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=2, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros(n_out, n_in + int(bias_x), n_in + int(bias_y))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        # s = s.permute(0, 2, 3, 1)

        return s


class LinearWithAct(nn.Module):
    def __init__(self, n_in, n_out, dropout=0) -> None:
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x

class GlobalPointer(nn.Module):
    def __init__(self, ent_type_size, hidden_size, inner_dim=64, RoPE=True):
        super().__init__()
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, x, attention_mask):
        self.device = x.device
        # x:(batch_size, seq_len, hidden_size)

        batch_size = x.size()[0]
        seq_len = x.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(x)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # # ??����?��?????�먨��?
        # mask = torch.tril(torch.ones_like(logits), -1)
        # logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

class PointerMatrix(nn.Module):

    def __init__(
        self,
        hidden_size,
        biaffine_size,
        cls_num=2,
        dropout=0,
        use_ldnet=True,
        biaffine_bias=False,
        use_rope=False,
    ):
        super().__init__()
        self.linear_h = LinearWithAct(
            n_in=hidden_size, n_out=biaffine_size, dropout=dropout
        )
        self.linear_t = LinearWithAct(
            n_in=hidden_size, n_out=biaffine_size, dropout=dropout
        )
        self.biaffine = Biaffine(
            n_in=biaffine_size,
            n_out=cls_num,
            bias_x=biaffine_bias,
            bias_y=biaffine_bias,
        )
        self.use_rope = use_rope
        self.use_ldnet = use_ldnet
        if self.use_ldnet:
            self.gp = GlobalPointer(cls_num, hidden_size)

    def sinusoidal_position_embedding(self, qw, kw):
        batch_size, seq_len, output_dim = qw.shape
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        pos_emb = position_ids * indices
        pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1)
        pos_emb = pos_emb.repeat((batch_size, *([1] * len(pos_emb.shape))))
        pos_emb = torch.reshape(pos_emb, (batch_size, seq_len, output_dim))
        pos_emb = pos_emb.to(qw)

        # (bs, seq_len, 1, hz) -> (bs, seq_len, hz)
        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        # (bs, seq_len, 1, hz) -> (bs, seq_len, hz)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.cat([-qw[..., 1::2], qw[..., ::2]], -1)
        qw = qw * cos_pos + qw2 * sin_pos
        kw2 = torch.cat([-kw[..., 1::2], kw[..., ::2]], -1)
        kw = kw * cos_pos + kw2 * sin_pos
        return qw, kw

    def forward(self, x, attention_mask=None):
        if self.use_ldnet:
            o = self.gp(x, attention_mask)
        else:
            h = self.linear_h(x)
            t = self.linear_t(x)
            if self.use_rope:
                h, t = self.sinusoidal_position_embedding(h, t)
            o = self.biaffine(h, t)
        return o

def multilabel_categorical_crossentropy(y_pred, y_true, bit_mask=None):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    if bit_mask is None:
        return neg_loss + pos_loss
    else:
        raise NotImplementedError


modeopt = {
    "name": "DynRT",
    "input1": "text",
    "input2": "img",
    "input3": "text_mask",
    "layer": 4,
    "tau_max": 10,
    "ORDERS": [
        0,
        1,
        2,
        3
    ],
    "IMG_SCALE": 7,
    "dropout": 0.5,
    "hidden_size": 1024,
    "ffn_size": 1024,
    "multihead": 2,
    "routing": "hard",
    "BINARIZE": False,
    "len": 256,
    "glimpses": 1,
    "mlp_size": 1024,
    "output_size": 1024,
    "orders": 4,
    "pooling": "avg",
    "classifier": "both",
    "roberta_path": "model/roberta-base",
    "roberta_layer": 1,
    "vitmodel": "vit_base_patch32_224",
    "finetune": False
}


class SchemaGuidedInstructBertModel(nn.Module):
    def __init__(
            self,
            plm_dir: str,
            bce_mean: bool = False,
            use_ldnet: bool = True,
            droprate: float = None,
            use_only_mr: bool = False,
            use_ldnet_ablation: bool = False,
            use_images: bool = True,
            vocab_size: int = None,
            use_rope: bool = True,
            biaffine_size: int = 512,
            label_mask_id: int = 4,
            text_mask_id: int = 7,
            dropout: float = 0.3,
    ):
        super().__init__()

        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  1     2   3           4    5   4    5   4    5   6    7    8   9          10    0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope
        self.use_images = use_images
        self.plm = AutoModel.from_pretrained(plm_dir)
        
        # self.plm = self.plm.to(torch.device("cuda"))
        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size

        self.use_ldnet = use_ldnet
        self.ablation = use_ldnet_ablation
        self.droprate = droprate
        self.use_only_mr = use_only_mr

        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            use_ldnet=self.use_ldnet,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

        self.image2token_emb = self.hidden_size
        self.max_seq = 256
        self.img_size = 1024
        self.up_hidden_size = 2048
        # self.model_resnet50 = timm.create_model('resnet101', pretrained=True, pretrained_cfg_overlay=dict(
        #     file='../NER/resnet101.a1h_in1k/pytorch_model.bin'))  # get the pre-trained ResNet model for the image
        # self.image_model = AutoModel.from_pretrained("Vim-small")
        # self.image_model = VisionMambaForImageClassification.from_pretrained("Vim-small")

        self.image_model = ViTModel.from_pretrained('vit-large-patch32-224-in21k')

        self.linear_extend_pic = nn.Linear(self.hidden_size, self.max_seq * self.image2token_emb)

        self.pic_mlp = nn.Sequential(
            nn.Linear(self.img_size, self.up_hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.up_hidden_size, self.up_hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.up_hidden_size, self.hidden_size),
        )

        # the attention mechanism for fine-grained features
        self.linear_q_fine = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k_fine = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v_fine = nn.Linear(self.hidden_size, self.hidden_size)

        # the attention mechanism for coarse-grained features
        self.linear_q_coarse = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_k_coarse = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear_v_coarse = nn.Linear(self.hidden_size, self.hidden_size)

        self.combine_linear = nn.Linear(self.hidden_size + self.image2token_emb * 2, self.hidden_size)
        self.combine_linear1 = nn.Linear(self.hidden_size + self.image2token_emb, self.hidden_size)

        # self.DynRT = DynRT_ED(modeopt)
        self.cls_layer = cls_layer_both(self.hidden_size, self.hidden_size)

        if self.use_ldnet:
            # ldnet
            self.bce = nn.BCEWithLogitsLoss(reduction='sum')
            self.bce_mean = nn.BCEWithLogitsLoss(reduction='mean')
            self.criterion = nn.BCEWithLogitsLoss(reduction='none') 
            self.sigmoid = nn.Sigmoid()
            self.cross_entropy = nn.CrossEntropyLoss()
            self.bce_linear = nn.Linear(self.hidden_size, 1)
            self.use_bce_mean_loss = bce_mean

    def input_encoding(self, input_ids, mask):

        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # out1=plm_outputs.last_hidden_state
        # out2=plm_outputs.last_hidden_state[:, -2, :]
        # out3=plm_outputs.hidden_states[-2]

        # bert_embed_text = self.bertl_text.embeddings(input_ids=input[self.input1])
        # for i in range(self.modelopt["roberta_layer"]):  # 1
        #     # don't understand
        #     bert_text = self.bertl_text.encoder.layer[i](bert_embed_text)[0]
        #     bert_embed_text = bert_text

        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        # _m = torch.logical_or(mask.eq(self.label_mask_id), mask.eq(self.text_mask_id))
        # mask_mat = _m.unsqueeze(-1).expand((bs, seq_len, seq_len))
        # # bit_mask: (batch_size, 1, seq_len, seq_len)
        # bit_mask = (
        #     torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        # )
        bit_mask = (
            mask.gt(0).unsqueeze(1).unsqueeze(1).expand(bs, 1, seq_len, seq_len).float()
        )

        return bit_mask

    def vit_forward(self, x):
        # x = self.image_model.patch_embed(x)
        # cls_token = self.image_model.cls_token.expand(x.shape[0], -1,
        #                                               -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x = torch.cat((cls_token, x), dim=1)
        # x = self.image_model.pos_drop(x + self.image_model.pos_embed)
        # x = self.image_model.blocks(x)
        # x = self.image_model.norm(x)
        # x = x[:, 1:]
        x = self.image_model(x)
        return x.last_hidden_state

    def forward(
            self, input_ids, mask, labels, images, bce_labels, bce_mask, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        scope = kwargs.get("scope")
        # bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)

        if self.use_ldnet:
            ## ldnet
            bce_logits = self.bce_linear(hidden).squeeze(dim=2) # [bs, seq_len] 
            losses = self.criterion(bce_logits, bce_labels)
            masked_losses = losses * bce_mask
            if self.use_bce_mean_loss:
                bce_loss = masked_losses.mean()
            else:
                bce_loss = masked_losses.sum()
            sigmoid_bce_logits = bce_logits
            ## ldnet

        # sigmoid_bce_logits = self.sigmoid(bce_logits)
        
        # lang_feat = []
        # lang_feat_mask = []
        # for i in range(hidden.size(0)):
        #     lang_feat.append(hidden[i, scope[i]:scope[i] + 128, :])
        #     lang_feat_mask.append(mask[i, scope[i]:scope[i] + 128])
        # lang_feat = torch.stack(lang_feat, dim=0)
        # lang_feat_mask = torch.stack(lang_feat_mask, dim=0).unsqueeze(1).unsqueeze(2).bool()

        # img_feat = self.vit_forward(images)
        # img_feat = self.linear_pic(img_feat)
        # img_feat_mask = torch.zeros([img_feat.shape[0], 1, 1, img_feat.shape[1]], dtype=torch.bool,
        #                             device=img_feat.device)
        #
        # hidden, img_feat = self.DynRT(hidden,
        #                                  img_feat,
        #                                  mask.unsqueeze(1).unsqueeze(2).bool(),
        #                                  img_feat_mask)
        if self.use_images:
            hidden = self.img_fusion2(hidden, images, scope)
        # img_feat=torch.sum(img_feat,dim=1)
        # img_feat=self.linear_extend_pic(img_feat).reshape(-1,self.max_seq,self.hidden_size)

        # alpha=torch.matmul(text_mean,img_mean)
        # for i in range(hidden.size(0)):
        #     hidden[i, scope[i]:scope[i] + 128, :] = lang_feat[i]



        # hidden=hidden+img_feat*0.5
        # (bs, 3, seq_len, seq_len)
        if self.use_ldnet or self.use_only_mr:
            logits = self.pointer(hidden, mask.gt(0).float())
        else:
            logits = self.pointer(hidden)

        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        assert labels.shape == (bs, cls_num, seq_len, seq_len)

        ### label drop
        if self.use_ldnet:
            label_drop = sigmoid_bce_logits.clone().reshape(bs, 1, 1, seq_len).repeat(1, cls_num, seq_len, 1)
            if self.ablation: 
                # import pdb
                # pdb.set_trace()
                wholenum = seq_len
                rownum = round(wholenum * self.droprate)
                selected_indices = torch.from_numpy(np.random.choice(wholenum, size=rownum, replace=False))
                selected_rows = logits[:, :, selected_indices, :]
                ldrop = sigmoid_bce_logits.clone().reshape(bs, 1, 1, seq_len).repeat(1, cls_num, len(selected_indices), 1)
                selected_rows = selected_rows * ldrop
                logits[:, :, selected_indices, :] = selected_rows
            else: 
                logits = logits * label_drop

        results = {"logits": logits}

        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            if self.use_ldnet:
                results["loss"] = loss + bce_loss
                results["bce_loss"] = bce_loss
            else:
                results["loss"] = loss

        if is_eval:
            batch_positions = self.decode(logits, labels, top_p=top_p, top_k=top_k, **kwargs)
            results["pred"] = batch_positions

        del loss, logits, hidden, bs, cls_num

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        return results

    def calc_path_prob(self, probs, paths):
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                path_prob *= probs[0, se[0], se[-1]]
            path_prob *= probs[1, path[-1], path[0]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor = None,
            top_p: float = 0.5,
            top_k: int = -1,
            # legal_num_parts: tuple = (1, 2, 3),
            legal_num_parts: tuple = None,
            **kwargs,
    ):
        # B x 3 x L x L
        # if labels is None:
        #     # `labels` is used for upper bound analysis
        #     probs = logits.sigmoid()
        #     pred = (probs > top_p).long()
        # else:
        #     pred = labels
        probs = logits.sigmoid()
        pred = (probs > top_p).long()
        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))
        # for pred, gold in zip(preds, kwargs.get("spans")):
        #     sorted_pred = sorted(set(tuple(x) for x in pred))
        #     sorted_gold = sorted(set(tuple(x) for x in gold))
        #     if sorted_pred != sorted_gold:
        #         breakpoint()

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        if legal_num_parts is not None:
            legal_preds = []
            for ins_paths in batch_preds:
                legal_paths = []
                for path in ins_paths:
                    if len(path) in legal_num_parts:
                        legal_paths.append(path)
                legal_preds.append(legal_paths)
        else:
            legal_preds = batch_preds

        return legal_preds

    def img_fusion(self, hidden, images, images_dif, aux_imgs, aux_images_dif, scope, weight, phrase_text):
        hidden_text = []
        for i in range(hidden.size(0)):
            hidden_text.append(hidden[i, scope[i]:scope[i] + 128, :])
        hidden_text = torch.stack(hidden_text, dim=0)

        feature_OriImg_FineGrained = self.model_resnet50.forward_features(images)
        feature_OriImg_CoarseGrained = self.model_resnet50.forward_features(aux_imgs.reshape(-1, 3, 224, 224))
        feature_DifImg_FineGrained = self.model_resnet50.forward_features(images_dif)
        feature_DifImg_CoarseGrained = self.model_resnet50.forward_features(aux_images_dif.reshape(-1, 3, 224, 224))

        pic_diff = torch.reshape(feature_DifImg_FineGrained, (-1, 2048, 49))
        pic_diff = torch.transpose(pic_diff, 1, 2)
        pic_diff = torch.reshape(pic_diff, (-1, 49, 2048))
        pic_diff = self.linear_pic(pic_diff)
        pic_diff_ = torch.sum(pic_diff, dim=1)

        pic_ori = torch.reshape(feature_OriImg_FineGrained, (-1, 2048, 49))
        pic_ori = torch.transpose(pic_ori, 1, 2)
        pic_ori = torch.reshape(pic_ori, (-1, 49, 2048))
        pic_ori = self.linear_pic(pic_ori)
        pic_ori_ = torch.sum(pic_ori, dim=1)

        pic_diff_objects = torch.reshape(feature_DifImg_CoarseGrained, (-1, 2048, 49))
        pic_diff_objects = torch.transpose(pic_diff_objects, 1, 2)
        pic_diff_objects = torch.reshape(pic_diff_objects, (-1, 3, 49, 2048))
        pic_diff_objects = torch.sum(pic_diff_objects, dim=2)
        pic_diff_objects = self.linear_pic(pic_diff_objects)
        pic_diff_objects_ = torch.sum(pic_diff_objects, dim=1)

        pic_ori_objects = torch.reshape(feature_OriImg_CoarseGrained, (-1, 2048, 49))
        pic_ori_objects = torch.transpose(pic_ori_objects, 1, 2)
        pic_ori_objects = torch.reshape(pic_ori_objects, (-1, 3, 49, 2048))
        pic_ori_objects = torch.sum(pic_ori_objects, dim=2)
        pic_ori_objects = self.linear_pic(pic_ori_objects)  # *weight_objects[:,:,0].reshape(-1,3,1)
        pic_ori_objects_ = torch.sum(pic_ori_objects, dim=1)  # .view(bsz, 16, 64)

        # output_text = self.bert(input_ids, attention_mask)
        # hidden_text = output_text['last_hidden_state']

        # output_phrases = self.bert(input_ids_phrase, attention_mask_phrase)
        # hidden_phrases = output_phrases['last_hidden_state']

        hidden_k_text = self.linear_k_fine(hidden_text)
        hidden_v_text = self.linear_v_fine(hidden_text)
        pic_q_diff = self.linear_q_fine(pic_diff)
        pic_diffusion = torch.sum(torch.tanh(self.att(pic_q_diff, hidden_k_text, hidden_v_text)), dim=1)

        hidden_k_text = self.linear_k_fine(hidden_text)
        hidden_v_text = self.linear_v_fine(hidden_text)
        pic_q_origin = self.linear_q_fine(pic_ori)
        pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)

        hidden_k_phrases = self.linear_k_coarse(hidden_text)
        hidden_v_phrases = self.linear_v_coarse(hidden_text)
        pic_q_diff_objects = self.linear_q_coarse(pic_diff_objects)
        pic_diffusion_objects = torch.sum(torch.tanh(self.att(pic_q_diff_objects, hidden_k_phrases, hidden_v_phrases)),
                                          dim=1)

        hidden_k_phrases = self.linear_k_coarse(hidden_text)
        hidden_v_phrases = self.linear_v_coarse(hidden_text)
        pic_q_ori_objects = self.linear_q_coarse(pic_ori_objects)
        pic_original_objects = torch.sum(torch.tanh(self.att(pic_q_ori_objects, hidden_k_phrases, hidden_v_phrases)),
                                         dim=1)

        # correlation allocation
        pic_ori_final = (pic_original + pic_ori_) * weight[:, 1].reshape(-1, 1) + (
                pic_original_objects + pic_ori_objects_) * weight[:, 0].reshape(-1, 1)
        pic_diff_final = (pic_diffusion + pic_diff_) * weight[:, 3].reshape(-1, 1) + (
                pic_diffusion_objects + pic_diff_objects_) * weight[:, 2].reshape(-1, 1)

        # assign image features to each token
        pic_ori = torch.tanh(self.linear_extend_pic(pic_ori_final).reshape(-1, self.max_seq, self.image2token_emb))
        pic_diff = torch.tanh(
            self.linear_extend_pic(pic_diff_final).reshape(-1, self.max_seq, self.image2token_emb))

        emissions = torch.relu(self.combine_linear(torch.cat([hidden_text, pic_ori, pic_diff], dim=-1)))
        for i in range(hidden.size(0)):
            hidden[i, scope[i]:scope[i] + 128, :] = emissions[i]
        return hidden

    def img_fusion1(self, hidden, images, images_dif, aux_imgs, aux_images_dif, scope, weight, phrase_text):
        hidden_text = []
        for i in range(hidden.size(0)):
            hidden_text.append(hidden[i, scope[i]:scope[i] + 128, :])
        hidden_text = torch.stack(hidden_text, dim=0)

        feature_OriImg_FineGrained = self.model_resnet50.forward_features(images)
        feature_OriImg_CoarseGrained = self.model_resnet50.forward_features(aux_imgs.reshape(-1, 3, 224, 224))
        # feature_DifImg_FineGrained = self.model_resnet50.forward_features(images_dif)
        # feature_DifImg_CoarseGrained = self.model_resnet50.forward_features(aux_images_dif.reshape(-1, 3, 224, 224))

        # pic_diff = torch.reshape(feature_DifImg_FineGrained, (-1, 2048, 49))
        # pic_diff = torch.transpose(pic_diff, 1, 2)
        # pic_diff = torch.reshape(pic_diff, (-1, 49, 2048))
        # pic_diff = self.linear_pic(pic_diff)
        # pic_diff_ = torch.sum(pic_diff, dim=1)

        pic_ori = torch.reshape(feature_OriImg_FineGrained, (-1, 2048, 49))
        pic_ori = torch.transpose(pic_ori, 1, 2)
        pic_ori = torch.reshape(pic_ori, (-1, 49, 2048))
        pic_ori = self.linear_pic(pic_ori)
        pic_ori_ = torch.sum(pic_ori, dim=1)

        # pic_diff_objects = torch.reshape(feature_DifImg_CoarseGrained, (-1, 2048, 49))
        # pic_diff_objects = torch.transpose(pic_diff_objects, 1, 2)
        # pic_diff_objects = torch.reshape(pic_diff_objects, (-1, 3, 49, 2048))
        # pic_diff_objects = torch.sum(pic_diff_objects, dim=2)
        # pic_diff_objects = self.linear_pic(pic_diff_objects)
        # pic_diff_objects_ = torch.sum(pic_diff_objects, dim=1)

        pic_ori_objects = torch.reshape(feature_OriImg_CoarseGrained, (-1, 2048, 49))
        pic_ori_objects = torch.transpose(pic_ori_objects, 1, 2)
        pic_ori_objects = torch.reshape(pic_ori_objects, (-1, 3, 49, 2048))
        pic_ori_objects = torch.sum(pic_ori_objects, dim=2)
        pic_ori_objects = self.linear_pic(pic_ori_objects)  # *weight_objects[:,:,0].reshape(-1,3,1)
        pic_ori_objects_ = torch.sum(pic_ori_objects, dim=1)  # .view(bsz, 16, 64)

        # output_text = self.bert(input_ids, attention_mask)
        # hidden_text = output_text['last_hidden_state']

        # output_phrases = self.bert(input_ids_phrase, attention_mask_phrase)
        # hidden_phrases = output_phrases['last_hidden_state']

        # hidden_k_text = self.linear_k_fine(hidden_text)
        # hidden_v_text = self.linear_v_fine(hidden_text)
        # pic_q_diff = self.linear_q_fine(pic_diff)
        # pic_diffusion = torch.sum(torch.tanh(self.att(pic_q_diff, hidden_k_text, hidden_v_text)), dim=1)

        hidden_k_text = self.linear_k_fine(hidden_text)
        hidden_v_text = self.linear_v_fine(hidden_text)
        pic_q_origin = self.linear_q_fine(pic_ori)
        pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_text, hidden_v_text)), dim=1)

        # hidden_k_phrases = self.linear_k_coarse(hidden_text)
        # hidden_v_phrases = self.linear_v_coarse(hidden_text)
        # pic_q_diff_objects = self.linear_q_coarse(pic_diff_objects)
        # pic_diffusion_objects = torch.sum(torch.tanh(self.att(pic_q_diff_objects, hidden_k_phrases, hidden_v_phrases)),
        #                                   dim=1)

        hidden_k_phrases = self.linear_k_coarse(hidden_text)
        hidden_v_phrases = self.linear_v_coarse(hidden_text)
        pic_q_ori_objects = self.linear_q_coarse(pic_ori_objects)
        pic_original_objects = torch.sum(torch.tanh(self.att(pic_q_ori_objects, hidden_k_phrases, hidden_v_phrases)),
                                         dim=1)

        # correlation allocation
        pic_ori_final = (pic_original + pic_ori_) * weight[:, 1].reshape(-1, 1) + (
                pic_original_objects + pic_ori_objects_) * weight[:, 0].reshape(-1, 1)
        # pic_diff_final = (pic_diffusion + pic_diff_) * weight[:, 3].reshape(-1, 1) + (
        #         pic_diffusion_objects + pic_diff_objects_) * weight[:, 2].reshape(-1, 1)

        # assign image features to each token
        pic_ori = torch.tanh(self.linear_extend_pic(pic_ori_final).reshape(-1, self.max_seq, self.image2token_emb))
        # pic_diff = torch.tanh(
        #     self.linear_extend_pic(pic_diff_final).reshape(-1, self.max_seq, self.image2token_emb))
        emissions = torch.relu(self.combine_linear1(torch.cat([hidden_text, pic_ori], dim=-1)))
        for i in range(hidden.size(0)):
            hidden[i, scope[i]:scope[i] + 128, :] = emissions[i]
        return hidden

    def img_fusion2(self, hidden, images, scope):
        # hidden_text = []
        # for i in range(hidden.size(0)):
        #     hidden_text.append(hidden[i, scope[i]:scope[i] + 128, :])
        # hidden_text = torch.stack(hidden_text, dim=0)

        feature_OriImg_FineGrained = self.vit_forward(images)

        pic_ori = self.pic_mlp(feature_OriImg_FineGrained)
        pic_ori_ = torch.sum(pic_ori, dim=1)

        # hidden_k_pic = self.linear_k_fine(pic_ori)
        # hidden_v_pic = self.linear_v_fine(pic_ori)
        # text_q_origin = self.linear_q_fine(hidden)
        # hidden_text = self.att(text_q_origin, hidden_k_pic, hidden_v_pic)

        hidden_k_hidden = self.linear_k_fine(hidden)
        hidden_v_hidden = self.linear_v_fine(hidden)
        pic_q_origin = self.linear_q_fine(pic_ori)
        pic_original = torch.sum(torch.tanh(self.att(pic_q_origin, hidden_k_hidden, hidden_v_hidden)), dim=1)


        # pic_ori_final = (pic_original + pic_ori_)
        # # assign image features to each token
        # pic_ori = torch.tanh(
        #     self.linear_extend_pic(pic_ori_).reshape(-1, self.max_seq, self.image2token_emb))
        pic_ori_final = torch.tanh(self.linear_extend_pic(pic_original).reshape(-1, self.max_seq, self.image2token_emb))
        #
        # # emissions = torch.relu(self.combine_linear1(torch.cat([hidden, pic_ori], dim=-1)))
        # emissions = hidden + pic_ori_final
        # for i in range(hidden.size(0)):
        #     hidden[i, scope[i]:scope[i] + 128, :] = emissions[i]
        return hidden+pic_ori_final*0.5

    # the attention mechanism
    def att(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)  # (5,50)
        att_map = F.softmax(scores, dim=-1)
        return torch.matmul(att_map, value)

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(sum=0.0, std=0.05)
