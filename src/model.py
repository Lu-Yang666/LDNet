import threading
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from rex.utils.iteration import windowed_queue_iter
from rex.utils.logging import logger
from transformers import AutoModel, BertModel, AutoModelForCausalLM
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, TaskType, PeftType

from src.utils import decode_nnw_nsw_thw_mat, decode_nnw_thw_mat, decode_pointer_mat
import pdb


class Biaffine(nn.Module):
    """Biaffine transformation

    References:
        - https://github.com/yzhangcs/parser/blob/main/supar/modules/affine.py
        - https://github.com/ljynlp/W2NER
    """

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

        # # 排除下三角
        # mask = torch.tril(torch.ones_like(logits), -1)
        # logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5

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


class PointerMatrix(nn.Module):
    """Pointer Matrix Prediction

    References:
        - https://github.com/ljynlp/W2NER
    """

    def __init__(
        self,
        hidden_size,
        biaffine_size,
        cls_num=2,
        dropout=0,
        biaffine_bias=False,
        use_rope=False,
    ):
        super().__init__()
        # self.linear_h = LinearWithAct(
        #     n_in=hidden_size, n_out=biaffine_size, dropout=dropout
        # )
        # self.linear_t = LinearWithAct(
        #     n_in=hidden_size, n_out=biaffine_size, dropout=dropout
        # )
        # self.biaffine = Biaffine(
        #     n_in=biaffine_size,
        #     n_out=cls_num,
        #     bias_x=biaffine_bias,
        #     bias_y=biaffine_bias,
        # )
        self.use_rope = use_rope
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

    # def forward(self, x):
    #     h = self.linear_h(x)
    #     t = self.linear_t(x)
    #     if self.use_rope:
    #         h, t = self.sinusoidal_position_embedding(h, t)
    #     o = self.biaffine(h, t)
    #     return o
    
    def forward(self, x, attention_mask):
        o = self.gp(x, attention_mask)
        return o

def multilabel_categorical_crossentropy(y_pred, y_true, bit_mask=None):
    """
    https://kexue.fm/archives/7359
    https://github.com/gaohongkui/GlobalPointer_pytorch/blob/main/common/utils.py
    """
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    ## remove nan
    # pos_min = torch.finfo(y_pred_pos.dtype).min
    # y_pred_pos[torch.isnan(y_pred_pos)] = pos_min 
    # print(y_pred_pos, y_pred_pos.shape, y_pred_pos.dtype)
    # print(torch.isnan(y_pred_pos).any())
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

    if bit_mask is None:
        return neg_loss + pos_loss
    else:
        raise NotImplementedError


class MrcPointerMatrixModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        cls_num: int = 2,
        biaffine_size: int = 384,
        none_type_id: int = 0,
        text_mask_id: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        # num of predicted classes, default is 3: None, NNW and THW
        self.cls_num = cls_num
        # None type id: 0, Next Neighboring Word (NNW): 1, Tail Head Word (THW): 2
        self.none_type_id = none_type_id
        # input: cls instruction sep text sep pad
        # mask:   1       2       3   4    5   0
        self.text_mask_id = text_mask_id

        self.plm = BertModel.from_pretrained(plm_dir)
        hidden_size = self.plm.config.hidden_size
        # self.biaffine_size = biaffine_size
        self.nnw_mat = PointerMatrix(
            hidden_size, biaffine_size, cls_num=2, dropout=dropout
        )
        self.thw_mat = PointerMatrix(
            hidden_size, biaffine_size, cls_num=2, dropout=dropout
        )
        self.criterion = nn.CrossEntropyLoss()

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        mask_mat = (
            mask.eq(self.text_mask_id).unsqueeze(-1).expand((bs, seq_len, seq_len))
        )
        # bit_mask: (batch_size, seq_len, seq_len, 1)
        bit_mask = (
            torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).long()
        )
        return bit_mask

    def forward(self, input_ids, mask, labels=None, is_eval=False, **kwargs):
        hidden = self.input_encoding(input_ids, mask)
        nnw_hidden = self.nnw_mat(hidden)
        thw_hidden = self.thw_mat(hidden)
        # nnw_hidden = nnw_hidden / self.biaffine_size ** 0.5
        # thw_hidden = thw_hidden / self.biaffine_size ** 0.5
        # # (bs, 2, seq_len, seq_len)
        bs, _, seq_len, seq_len = nnw_hidden.shape

        bit_mask = self.build_bit_mask(mask)

        results = {"logits": {"nnw": nnw_hidden, "thw": thw_hidden}}
        if labels is not None:
            # mean
            nnw_loss = self.criterion(
                nnw_hidden.permute(0, 2, 3, 1).reshape(-1, 2),
                labels[:, 0, :, :].reshape(-1),
            )
            thw_loss = self.criterion(
                thw_hidden.permute(0, 2, 3, 1).reshape(-1, 2),
                labels[:, 1, :, :].reshape(-1),
            )
            loss = nnw_loss + thw_loss
            results["loss"] = loss

        if is_eval:
            batch_positions = self.decode(nnw_hidden, thw_hidden, bit_mask, **kwargs)
            results["pred"] = batch_positions
        return results

    def decode(
        self,
        nnw_hidden: torch.Tensor,
        thw_hidden: torch.Tensor,
        bit_mask: torch.Tensor,
        **kwargs,
    ):
        # B x L x L
        nnw_pred = nnw_hidden.argmax(1)
        thw_pred = thw_hidden.argmax(1)
        # B x 2 x L x L
        pred = torch.stack([nnw_pred, thw_pred], dim=1)
        pred = pred * bit_mask

        batch_preds = decode_nnw_thw_mat(pred, offsets=kwargs.get("offset"))

        return batch_preds


class MrcGlobalPointerModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        use_rope: bool = True,
        cls_num: int = 2,
        biaffine_size: int = 384,
        none_type_id: int = 0,
        text_mask_id: int = 4,
        dropout: float = 0.3,
        mode: str = "w2",
    ):
        super().__init__()

        # num of predicted classes, default is 3: None, NNW and THW
        self.cls_num = cls_num
        # None type id: 0, Next Neighboring Word (NNW): 1, Tail Head Word (THW): 2
        self.none_type_id = none_type_id
        # input: cls instruction sep text sep pad
        # mask:   1       2       3   4    5   0
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        # mode: w2: w2ner, cons: consecutive spans
        self.mode = mode
        assert self.mode in ["w2", "cons"]

        self.plm = BertModel.from_pretrained(plm_dir)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=2 if self.mode == "w2" else 1,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state

    def build_bit_mask(self, mask: torch.Tensor) -> torch.Tensor:
        # mask: (batch_size, seq_len)
        bs, seq_len = mask.shape
        mask_mat = (
            mask.eq(self.text_mask_id).unsqueeze(-1).expand((bs, seq_len, seq_len))
        )
        # bit_mask: (batch_size, 1, seq_len, seq_len)
        bit_mask = (
            torch.logical_and(mask_mat, mask_mat.transpose(1, 2)).unsqueeze(1).float()
        )
        if self.mode == "cons":
            bit_mask = bit_mask.triu()

        return bit_mask

    def forward(
        self, input_ids, mask, labels=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        bit_mask = self.build_bit_mask(mask)
        hidden = self.input_encoding(input_ids, mask)
        # (bs, 2, seq_len, seq_len)
        logits = self.pointer(hidden)
        logits = logits * bit_mask - (1.0 - bit_mask) * 1e12
        logits = logits / (self.biaffine_size**0.5)
        # # (bs, 2, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        assert labels.shape == (bs, cls_num, seq_len, seq_len)

        results = {"logits": logits}
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            results["loss"] = loss

        if is_eval:
            batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            results["pred"] = batch_positions
        return results

    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
        assert self.mode in ["w2", "cons"]
        paths_with_prob = []
        for path in paths:
            path_prob = 1.0
            if self.mode == "w2":
                for se in windowed_queue_iter(path, 2, 1, drop_last=True):
                    path_prob *= probs[0, se[0], se[-1]]
                path_prob *= probs[1, path[-1], path[0]]
            elif self.mode == "cons":
                path_prob = probs[0, path[0], path[-1]]
            paths_with_prob.append((path, path_prob))
        return paths_with_prob

    def decode(
        self,
        logits: torch.Tensor,
        top_p: float = 0.5,
        top_k: int = -1,
        **kwargs,
    ):
        # mode: w2: w2ner with nnw and thw labels, cons: consecutive spans with one type of labels
        assert self.mode in ["w2", "cons"]
        # B x 2 x L x L
        probs = logits.sigmoid()
        pred = (probs > top_p).long()
        if self.mode == "w2":
            preds = decode_nnw_thw_mat(pred, offsets=kwargs.get("offset"))
        elif self.mode == "cons":
            pred = pred.triu()
            preds = decode_pointer_mat(pred, offsets=kwargs.get("offset"))

        if top_k == -1:
            batch_preds = preds
        else:
            batch_preds = []
            for i, paths in enumerate(preds):
                paths_with_prob = self.calc_path_prob(probs[i], paths)
                paths_with_prob.sort(key=lambda pp: pp[1], reverse=True)
                batch_preds.append([pp[0] for pp in paths_with_prob[:top_k]])

        return batch_preds


class SchemaGuidedInstructBertModel(nn.Module):
    def __init__(
        self,
        plm_dir: str,
        vocab_size: int = None,
        use_rope: bool = True,
        biaffine_size: int = 512,
        label_mask_id: int = 4,
        text_mask_id: int = 7,
        dropout: float = 0.3,
        kd: bool = False,
        generate_logits: bool = False,
        teacher_logits: Optional[dict] = None,
        fewshot: bool = False,
        ablation: bool = False,
        droprate: float = 0.1,
        lora: bool = False,
        epoch: int = None,
        only_use_mr: bool = False
    ):
        super().__init__()

        # input: [CLS] [I] Instruction [LM] PER [LM] LOC [LM] ORG [TL] Text [B] Background [SEP] [PAD]
        # mask:  1     2   3           4    5   4    5   4    5   6    7    8   9          10    0
        self.label_mask_id = label_mask_id
        self.text_mask_id = text_mask_id
        self.use_rope = use_rope

        self.lora = lora
        if self.lora:
            model = AutoModel.from_pretrained(plm_dir)
            self.plm = self.load_model(model)
            # self.plm = self.load_ptuning_model(plm_dir)
            # self.plm = AutoModel.from_pretrained(plm_dir, trust_remote_code=True)

            ### freeze embedding model
            # self.plm = AutoModel.from_pretrained(plm_dir, trust_remote_code=True)
            # for param in self.plm.parameters():
            #     param.requires_grad = False
        else:
            self.plm = AutoModel.from_pretrained(plm_dir)

        if vocab_size:
            self.plm.resize_token_embeddings(vocab_size)
        self.hidden_size = self.plm.config.hidden_size
        self.biaffine_size = biaffine_size
        self.pointer = PointerMatrix(
            self.hidden_size,
            biaffine_size,
            cls_num=3,
            dropout=dropout,
            biaffine_bias=True,
            use_rope=use_rope,
        )
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.bce_mean = nn.BCEWithLogitsLoss(reduction='mean')
        self.criterion = nn.BCEWithLogitsLoss(reduction='none') 
        # self.bce = nn.BCELoss(reduction='sum')
        # self.bce_mean = nn.BCELoss(reduction='mean')        
        self.kd = kd
        self.teacher_logits = teacher_logits
        self.generate_logits = generate_logits
        self.fewshot = fewshot
        self.sigmoid = nn.Sigmoid()
        self.mse = nn.MSELoss()
        self.ablation = ablation
        self.droprate = droprate
        self.bce_linear = nn.Linear(self.hidden_size, 1)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.only_use_mr = only_use_mr

    def find_all_linear_names(self, model):
        """
        找出所有全连接层，为所有全连接添加adapter
        """
        cls = nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        lora_module_names = list(lora_module_names)
        logger.info(f'LoRA target module names: {lora_module_names}')
        return lora_module_names

    def load_model(self, model):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        target_modules = self.find_all_linear_names(model)

        lora_rank = 64
        lora_alpha = 16
        lora_dropout = 0.05

        peft_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            # task_type="CAUSAL_LM",
        )
        # pdb.set_trace()
        model = get_peft_model(model, peft_config)
        logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        model.print_trainable_parameters()
        return model.model

    def load_ptuning_model(self, model_name_or_path):
        peft_config = PrefixTuningConfig(task_type=TaskType.FEATURE_EXTRACTION, num_virtual_tokens=20)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def input_encoding(self, input_ids, mask):
        attention_mask = mask.gt(0).float()
        plm_outputs = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return plm_outputs.last_hidden_state, attention_mask

    def forward(
        self, input_ids, mask, bce_labels, bce_mask, labels=None, raw=None, spans=None, is_eval=False, top_p=0.5, top_k=-1, **kwargs
    ):
        hidden, attention_mask = self.input_encoding(input_ids, mask)
        bce_logits = self.bce_linear(hidden).squeeze(dim=2) # [bs, seq_len] 
        losses = self.criterion(bce_logits, bce_labels)
        masked_losses = losses * bce_mask
        bce_loss = masked_losses.mean()
        # sigmoid_bce_logits = bce_logits
        sigmoid_bce_logits = self.sigmoid(bce_logits)

        # (bs, 3, seq_len, seq_len)
        logits = self.pointer(hidden, attention_mask)
        # # (bs, 3, seq_len, seq_len)
        bs, cls_num, seq_len, seq_len = logits.shape
        
        ### label drop
        if not self.only_use_mr:
            label_drop = sigmoid_bce_logits.clone().reshape(bs, 1, 1, seq_len).repeat(1, cls_num, seq_len, 1)
            if self.ablation: 
                wholenum = seq_len
                rownum = round(wholenum * self.droprate)
                selected_indices = torch.from_numpy(np.random.choice(wholenum, size=rownum, replace=False))
                selected_rows = logits[:, :, selected_indices, :]
                ldrop = sigmoid_bce_logits.clone().reshape(bs, 1, 1, seq_len).repeat(1, cls_num, len(selected_indices), 1)
                selected_rows = selected_rows * ldrop
                logits[:, :, selected_indices, :] = selected_rows
            else: 
                logits = logits * label_drop
        assert labels.shape == (bs, cls_num, seq_len, seq_len)

        results = {"logits": logits}
        # results["bce_label"] = masked_target
        # results["bce_preds"] = masked_input
        if labels is not None:
            loss = multilabel_categorical_crossentropy(
                logits.reshape(bs * cls_num, -1), labels.reshape(bs * cls_num, -1)
            )
            loss = loss.mean()
            ### model transfer learning
            if self.kd == True: 
                for i in range(bs):
                    k = raw[i]['id']
                    if k in self.teacher_logits.keys():
                        bs, oseqlen, _ = self.teacher_logits[k].shape
                        loss += self.mse(logits[i][:, :oseqlen, :oseqlen], self.teacher_logits[k])
            if not self.only_use_mr:
                results["loss"] = loss + bce_loss
            else:
                results["loss"] = loss
            results["bce_loss"] = bce_loss
            # results["bce_loss_mean"] = bce_mean_loss

        if is_eval:
            batch_positions = self.decode(logits, top_p=top_p, top_k=top_k, **kwargs)
            results["pred"] = batch_positions

        return results
    
    def calc_path_prob(self, probs, paths):
        """
        Args:
            probs: (2, seq_len, seq_len) | (1, seq_len, seq_len)
            paths: a list of paths in tuple

        Returns:
            [(path: tuple, prob: float), ...]
        """
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
        top_p: float = 0.5,
        top_k: int = -1,
        spans: List = None,
        # legal_num_parts: tuple = (1, 2, 3),
        legal_num_parts: tuple = None,
        labels: torch.Tensor = None,
        **kwargs,
    ):
        # B x 3 x L x L
        if labels is None:
            # `labels` is used for upper bound analysis
            probs = logits.sigmoid()
            pred = (probs > top_p).long()
        else:
            pred = labels

        preds = decode_nnw_nsw_thw_mat(pred, offsets=kwargs.get("offset"))

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
