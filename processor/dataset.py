import random
import os
from collections import defaultdict

import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BatchEncoding, DebertaV2TokenizerFast,AutoTokenizer
from torchvision import transforms
import logging
import json
from utils import (
    decode_nnw_nsw_thw_mat,
    decode_nnw_thw_mat,
    encode_nnw_nsw_thw_mat,
    encode_nnw_thw_mat,
)

logger = logging.getLogger(__name__)


class Mirrorprocessor(object):
    def __init__(self, text_path, encoder_name, use_llm, use_qwen) -> None:
        self.text_path = text_path
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, do_lower_case=True, trust_remote_code=True)

        self.lc_token = "[LC]"
        self.lm_token = "[LM]"
        self.lr_token = "[LR]"
        self.i_token = "[I]"
        self.tl_token = "[TL]"
        self.tp_token = "[TP]"
        self.b_token = "[B]"
        # for mre
        self.head_token = "[HEAD]"
        self.tail_token = "[TAIL]"
        self.head_token1 = "[/HEAD]"
        self.tail_token1 = "[/TAIL]"

        if use_llm:
            logger.info("add special tokens")
            cls_token = "[CLS]"
            sep_token = "[SEP]"
            self.tokenizer.add_tokens([cls_token, sep_token])
            self.tokenizer.add_special_tokens({"cls_token": cls_token, "sep_token": sep_token})
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = 'right'
            self.tokenizer.truncation_side = 'right'
            # pass

        if not use_qwen:
            logger.info("add special tokens")
            num_added = self.tokenizer.add_tokens(
                [
                    self.lc_token,
                    self.lm_token,
                    self.lr_token,
                    self.i_token,
                    self.tl_token,
                    self.tp_token,
                    self.b_token,
                ]
            )
            assert num_added == 7
            
        
    def load_from_file(self, mode="train", sample_ratio=1.0):
        """
        mode: dataset mode. Defaults to "train"
        sample_ratio: sample ratio in low resouce. Defaults to 1.0
        """
        load_file = os.path.join(self.text_path,f"{mode}.jsonl")
        logger.info("Loading data from {}".format(load_file))
        instance = []
        # data_dict = defaultdict(dict)
        # split text and img id
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

            for line in lines:
                json_object = json.loads(line)
                instance.append(json_object)

        # for dataset in {"twitter15", "twitter17", "mre"}:
        #     for mode in {'train', 'dev', 'test'}:
        #         # load visual objects from original and generated images
        #         # aux_path = self.data_path[dataset][mode + "_auximgs"]
        #         # aux_imgs = torch.load(aux_path)
        #         # aux_path_dif = self.data_path[dataset][mode + "_auximgs_dif"]
        #         # aux_imgs_dif = torch.load(aux_path_dif)
        #
        #         # load weak correlation between text and original image
        #         with open(self.data_path[dataset]['%s_weak_ori' % mode], 'r', encoding='utf-8') as f_rel:
        #             lines = f_rel.readlines()
        #             weak_ori = {}
        #             for line in lines:
        #                 img_id_key, score = line.split('	')[0], float(line.split('	')[1].replace('\n', ''))
        #                 weak_ori[img_id_key] = score
        #
        #         # load strong correlation between text and original image
        #         with open(self.data_path[dataset]['%s_strong_ori' % mode], 'r', encoding='utf-8') as f_rel:
        #             lines = f_rel.readlines()
        #             strong_ori = {}
        #             for line in lines:
        #                 img_id_key, score = line.split('	')[0], float(line.split('	')[1].replace('\n', ''))
        #                 strong_ori[img_id_key] = score
        #
        #         # load weak correlation between text and generated image
        #         with open(self.data_path[dataset]['%s_weak_dif' % mode], 'r', encoding='utf-8') as f_rel:
        #             lines = f_rel.readlines()
        #             weak_dif = {}
        #             for line in lines:
        #                 img_id_key, score = line.split('	')[0], float(line.split('	')[1].replace('\n', ''))
        #                 weak_dif[img_id_key] = score
        #
        #         # load strong correlation between text and generated image
        #         with open(self.data_path[dataset]['%s_strong_dif' % mode], 'r', encoding='utf-8') as f_rel:
        #             lines = f_rel.readlines()
        #             strong_dif = {}
        #             for line in lines:
        #                 img_id_key, score = line.split('	')[0], float(line.split('	')[1].replace('\n', ''))
        #                 strong_dif[img_id_key] = score
        #
        #         # load phrases for visual objects detection
        #         with open(self.data_path[dataset]['%s_grounding_text' % mode], 'r',
        #                   encoding='utf-8') as f_ner_phrase_text:
        #             data_phrase_text = json.load(f_ner_phrase_text)
        #         data_dict[dataset][mode] = {"aux_imgs": aux_imgs, 'aux_imgs_dif': aux_imgs_dif,
        #                                     'weak_ori': weak_ori, 'strong_ori': strong_ori,
        #                                     'weak_dif': weak_dif, 'strong_dif': strong_dif,
        #                                     'phrase_text': data_phrase_text}
        # sample data, only for low-resource
        if sample_ratio != 1.0:
            # 计算要采样的样本数量
            num_samples = int(len(instance) * sample_ratio)

            # 随机选择要保留的样本索引
            sampled_indices = random.sample(range(len(instance)), num_samples)

            # 根据采样的索引构建采样后的实例列表
            sampled_instance = [instance[i] for i in sampled_indices]

            # 返回采样后的实例列表作为结果
            return sampled_instance

        return instance


class MulMirrorDataset(Dataset):
    def __init__(self, processor, use_llm: bool = False, use_qwen: bool = False, label_span: str = "tag", include_instructions: bool = True,
                 max_seq=256, sample_ratio=1, mode="train", ignore_idx=0) -> None:
        self.processor = processor
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        self.instance = processor.load_from_file(mode, sample_ratio)
        self.tokenizer = processor.tokenizer
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        if use_llm:
            # self.lc_token = "<|extra_0|>"
            # self.lm_token = "<|extra_1|>"
            # self.lr_token = "<|extra_2|>"
            # self.i_token = "<|extra_3|>"
            # self.tl_token = "<|extra_4|>"
            # self.tp_token = "<|extra_5|>"
            # self.b_token = "<|extra_6|>"
            self.lc_token = "[LC]"
            self.lm_token = "[LM]"
            self.lr_token = "[LR]"
            self.i_token = "[I]"
            self.tl_token = "[TL]"
            self.tp_token = "[TP]"
            self.b_token = "[B]"
        elif use_qwen:
            self.lc_token = "1"
            self.lm_token = "2"
            self.lr_token = "3"
            self.i_token = "4"
            self.tl_token = "5"
            self.tp_token = "6"
            self.b_token = "7"
        else:
            self.lc_token = "[LC]"
            self.lm_token = "[LM]"
            self.lr_token = "[LR]"
            self.i_token = "[I]"
            self.tl_token = "[TL]"
            self.tp_token = "[TP]"
            self.b_token = "[B]"
        self.label_span = label_span
        self.include_instructions = include_instructions

        self.mode = mode
        self.sample_ratio = sample_ratio

        self.use_llm = use_llm
        self.use_qwen = use_qwen

    def __len__(self):
        return len(self.instance)

    def __getitem__(self, idx):
        instance = self.instance[idx]
        # picture_transform
        id = instance["id"]
        img = instance["img_id"]
        if id.startswith("NER"):
            dataset = id.split(".")[1]
            mode = id.split(".")[2]
        else:
            dataset = "mre"
            mode = id.split(".")[1].split("_")[1]
        image = self.pic_transform(dataset, mode, img)

        # input
        if self.use_llm:
            tokens = [self.tokenizer.cls_token]
            mask = [1]
        elif self.use_qwen:
            tokens=[]
            mask=[]
        else:
            tokens = [self.tokenizer.cls_token]
            mask = [1]
        label_map = {"lc": {}, "lm": {}, "lr": {}}
        # (2, 3): {"type": "lc", "task": "cls/ent/rel/event/hyper_rel/discontinuous_ent", "string": ""}
        span_to_label = {}

        def _update_seq(
                label: str,
                label_type: str,
                task: str = "",
                label_mask: int = 4,
                content_mask: int = 5,
        ):
            if label not in label_map[label_type]:
                label_token_map = {
                    "lc": self.lc_token,
                    "lm": self.lm_token,
                    "lr": self.lr_token,
                }
                label_tag_start_idx = len(tokens)
                tokens.append(label_token_map[label_type])
                mask.append(label_mask)
                label_tag_end_idx = len(tokens) - 1  # exact end position
                label_tokens = self.tokenizer(label, add_special_tokens=False).tokens()
                label_content_start_idx = len(tokens)
                tokens.extend(label_tokens)
                mask.extend([content_mask] * len(label_tokens))
                label_content_end_idx = len(tokens) - 1  # exact end position

                if self.label_span == "tag":
                    start_idx = label_tag_start_idx
                    end_idx = label_tag_end_idx
                elif self.label_span == "content":
                    start_idx = label_content_start_idx
                    end_idx = label_content_end_idx
                else:
                    raise ValueError(f"label_span={self.label_span} is not supported")

                if end_idx == start_idx:
                    label_map[label_type][label] = (start_idx,)
                else:
                    label_map[label_type][label] = (start_idx, end_idx)
                span_to_label[label_map[label_type][label]] = {
                    "type": label_type,
                    "task": task,
                    "string": label,
                }
            return label_map[label_type][label]

        if self.include_instructions:
            instruction = instance.get("instruction")
            if not instruction:
                logger.warning(
                    "include_instructions=True, while the instruction is empty!"
                )
        else:
            instruction = ""
        if instruction:
            tokens.append(self.i_token)
            mask.append(2)
            instruction_tokens = self.tokenizer(
                instruction, add_special_tokens=False
            ).tokens()
            tokens.extend(instruction_tokens)
            mask.extend([3] * len(instruction_tokens))
        types = instance["schema"].get("cls")
        if types:
            for t in types:
                _update_seq(t, "lc", task="cls")
        mention_types = instance["schema"].get("ent")
        if mention_types:
            for mt in mention_types:
                _update_seq(mt, "lm", task="ent")
        discon_ent_types = instance["schema"].get("discontinuous_ent")
        if discon_ent_types:
            for mt in discon_ent_types:
                _update_seq(mt, "lm", task="discontinuous_ent")
        rel_types = instance["schema"].get("rel")
        if rel_types:
            for rt in rel_types:
                _update_seq(rt, "lr", task="rel")
        hyper_rel_schema = instance["schema"].get("hyper_rel")
        if hyper_rel_schema:
            for rel, qualifiers in hyper_rel_schema.items():
                _update_seq(rel, "lr", task="hyper_rel")
                for qualifier in qualifiers:
                    _update_seq(qualifier, "lr", task="hyper_rel")
        event_schema = instance["schema"].get("event")
        if event_schema:
            for event_type, roles in event_schema.items():
                _update_seq(event_type, "lm", task="event")
                for role in roles:
                    _update_seq(role, "lr", task="event")

        text = instance.get("text")
        mrc_tp_index = 0
        rev = None
        if text:
            text_tokenized = self.tokenizer(
                text, return_offsets_mapping=True, add_special_tokens=False, max_length=100,
                truncation=True, padding="max_length"
            )
            if any(val for val in label_map.values()):
                text_label_token = self.tl_token
            else:
                text_label_token = self.tp_token
            tokens.append(text_label_token)
            mrc_tp_index = len(tokens) - 1
            mask.append(6)

            remain_token_len = self.max_seq - 1 - len(tokens)
            if remain_token_len < 100:
                return None
            text_off = len(tokens)
            text_tokens = text_tokenized.tokens()[:remain_token_len]
            tokens.extend(text_tokens)
            mask.extend([7] * sum(text_tokenized["attention_mask"]))
            # special token ----padding
            mask.extend([0] * (len(text_tokens) - sum(text_tokenized["attention_mask"])))
        else:
            text_tokenized = None

        if self.use_llm:
            # pass
            tokens.append(self.tokenizer.sep_token)
            mask.append(10)
        elif self.use_qwen:
            pass
        else:
            tokens.append(self.tokenizer.sep_token)
            mask.append(10)

        # labels
        # spans: [[(ent_type start, ent_type end + 1), (ent s, ent e + 1)]]
        spans = []  # one span may have many parts
        bce_label_spans = []
        if "cls" in instance["ans"]:
            for t in instance["ans"]["cls"]:
                part = label_map["lc"][t]
                spans.append([part])

        if "discontinuous_ent" in instance["ans"]:
            for ent in instance["ans"]["discontinuous_ent"]:
                label_part = label_map["lm"][ent["type"]]
                ent_span = [label_part]
                for part in ent["span"]:
                    position_seq = self.char_to_token_span(
                        part, text_tokenized, text_off
                    )
                    ent_span.append(position_seq)
                spans.append(ent_span)
        if "rel" in instance["ans"]:
            for rel in instance["ans"]["rel"]:
                label_part = label_map["lr"][rel["relation"]]
                if rev == False:
                    rel["head"]["span"] = [ele + 6 for ele in rel["head"]["span"]]
                    rel["tail"]["span"] = [ele + 19 for ele in rel["tail"]["span"]]
                elif rev == True:
                    rel["head"]["span"] = [ele + 19 for ele in rel["head"]["span"]]
                    rel["tail"]["span"] = [ele + 6 for ele in rel["tail"]["span"]]
                head_position_seq = self.char_to_token_span(
                    rel["head"]["span"], text_tokenized, text_off
                )
                tail_position_seq = self.char_to_token_span(
                    rel["tail"]["span"], text_tokenized, text_off
                )
                spans.append([label_part, head_position_seq, tail_position_seq])

        if "ent" in instance["ans"]:
            for ent in instance["ans"]["ent"]:
                label_part = label_map["lm"][ent["type"]]
                position_seq = self.char_to_token_span(
                    ent["span"], text_tokenized, text_off
                )
                if rev != None:
                    if ent["text"] == instance["ans"]["rel"][0]["head"]["text"]:
                        position_seq = head_position_seq
                    elif ent["text"] == instance["ans"]["rel"][0]["tail"]["text"]:
                        position_seq = tail_position_seq

                spans.append([label_part, position_seq])

        if "event" in instance["ans"]:
            for event in instance["ans"]["event"]:
                event_type_label_part = label_map["lm"][event["event_type"]]
                trigger_position_seq = self.char_to_token_span(
                    event["trigger"]["span"], text_tokenized, text_off
                )
                trigger_part = [event_type_label_part, trigger_position_seq]
                spans.append(trigger_part)
                for arg in event["args"]:
                    role_label_part = label_map["lr"][arg["role"]]
                    arg_position_seq = self.char_to_token_span(
                        arg["span"], text_tokenized, text_off
                    )
                    arg_part = [role_label_part, trigger_position_seq, arg_position_seq]
                    spans.append(arg_part)
        if "span" in instance["ans"]:
            # Extractive-QA or Extractive-MRC tasks
            for span in instance["ans"]["span"]:
                span_position_seq = self.char_to_token_span(
                    span["span"], text_tokenized, text_off
                )
                spans.append([span_position_seq])

        schema_label_spans = []
        if "span" in instance["ans"]:
            # if len(instance["ans"]["span"]) > 1:
            #     print(instance["ans"])
            #     for span in instance["ans"]["span"]:
            #         assert span["text"] != "ANSWERNOTFOUND"
            # Extractive-QA or Extractive-MRC tasks
            for span in instance["ans"]["span"]:
                span_position_seq = self.char_to_token_span(
                    span["span"], text_tokenized, text_off
                )
                spans.append([span_position_seq])
                if span["text"] == "ANSWERNOTFOUND":
                    schema_label_spans.append(mrc_tp_index)
                else: 
                    bce_label_spans.append(mrc_tp_index)

        bce_labels = [0]*len(tokens)
        # print(instance["ans"])
        # print(spans)
        # print(len(tokens))
        for index in bce_label_spans:
            bce_labels[index] = 1
        for span in spans:
            if len(span) == 1:
                if len(span[0]) == 1:
                    if span[0][0] >= 512: continue
                    bce_labels[span[0][0]] = 1
                else:
                    if span[0][0]>=512 or span[0][1]>512: continue
                    if span[0][1] == 512:
                        bce_labels[span[0][0]:] = [1]*len(bce_labels[span[0][0]:])
                    else:
                        bce_labels[span[0][0]:span[0][1]+1] = [1]*len(bce_labels[span[0][0]:span[0][1]+1])
            else:
                for s in span:
                    if len(s) == 1:
                        if s[0] >= 512: continue
                        bce_labels[s[0]] = 1
                    else:
                        if s[0]>=512 or s[1]>512: continue
                        if s[1]==512:
                            bce_labels[s[0]:] = [1]*len(bce_labels[s[0]:])
                        else:
                            bce_labels[s[0]:s[1]+1] = [1]*len(bce_labels[s[0]:s[1]+1])

        if len(tokens) < self.max_seq:
            padding_length = self.max_seq - len(tokens)
            tokens += [self.tokenizer.pad_token] * padding_length
            mask += [0] * padding_length
            bce_labels += [0] * padding_length
        bce_mask = [True] * len(tokens)
        for i, token in enumerate(tokens):
            if token == self.tokenizer.pad_token:
                bce_mask[i] = False
            # print(self.tokenizer.convert_tokens_to_ids(token))
        # print(tokens)
        # print(mask)
        
        ins = {
            "raw": instance,
            "tokens": tokens,
            "input_ids": torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens)),
            "mask": torch.tensor(mask),
            "spans": spans,
            "bce_labels": torch.tensor(bce_labels, dtype=torch.float),
            "bce_mask": torch.tensor(bce_mask, dtype=torch.bool),
            "label_map": label_map,
            "span_to_label": span_to_label,
            "labels": torch.tensor(encode_nnw_nsw_thw_mat(spans, self.max_seq)),
            # labels are calculated dynamically in collate_fn
            "img": image,
            "scope": text_off,
        }
        return ins

    def char_to_token_span(
            self, span: list[int], tokenized: BatchEncoding, offset: int = 0
    ) -> list[int]:
        token_s = tokenized.char_to_token(span[0])
        token_e = tokenized.char_to_token(span[1] - 1)
        if token_e == token_s:
            position_seq = (offset + token_s,)
        else:
            position_seq = (offset + token_s, offset + token_e)
        return position_seq

    def pic_transform(self, dataset: str, mode, img):
        if dataset == "twitter2015":
            # img_path = "twitter15_data/twitter2015_images"
            img_path = "/data/share2/strawberry/MM-LDNet/NER/data/twitter2015_images"
        elif dataset == "twitter2017":
            # img_path = "data/twitter2017_images"
            img_path = "/data/share2/strawberry/MM-LDNet/NER/data/twitter2017_images"
        elif dataset == "mre":
            # img_path = f"RE/data/img_org/{mode}"
            img_path = f"/data/share2/strawberry/MM-LDNet/RE/data/img_org/{mode}"
        else:
            img_path = None

        # image process
        if img_path is not None:
            # fine-grained image feature processing
            img_path = os.path.join(img_path, img)
            try:
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                # if the image doesn't exist, use all zero tensors for substitution
                image = torch.zeros(3,224,224)
                #print(img_path)

        return image

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    mask = [item["mask"] for item in batch]
    images = [item["img"] for item in batch]
    labels = [item["labels"] for item in batch]
    bce_labels = [item["bce_labels"] for item in batch]
    bce_mask = [item["bce_mask"] for item in batch]
    input_ids = torch.stack(input_ids)
    mask = torch.stack(mask)
    images = torch.stack(images)
    labels = torch.stack(labels)
    bce_labels = torch.stack(bce_labels)
    bce_mask = torch.stack(bce_mask)
    other_data = {key: [item[key] for item in batch] for key in batch[0] if key not in ["input_ids", "mask", "img", "labels", "bce_labels", "bce_mask"]}
    return {
        "input_ids": input_ids,
        "mask": mask,
        "labels":labels,
        "images": images,
        "bce_labels": bce_labels,
        "bce_mask": bce_mask,
        **other_data
    }


if __name__ == "__main__":
    path = {"train": "../data/newInst/RE/data/mre/test.jsonl"}
    model = "../deberta-v3-large"
    processor = Mirrorprocessor(path, model)
    dataset = MulMirrorDataset(processor)
    train_loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    for batch in train_loader:
        batch_id = batch

