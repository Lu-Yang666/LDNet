import datetime
import random
import string
from collections import defaultdict
from typing import Tuple, Union, Optional, Dict, List

import numpy as np

from sklearn.metrics import accuracy_score, matthews_corrcoef

DEFAULT_PRF1_RESULT_DICT = {"p": 0.0, "r": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}


def safe_division(
        numerator: Union[int, float], denominator: Union[int, float]
) -> float:
    try:
        val = numerator / denominator
    except ZeroDivisionError:
        val = 0.0
    return val


def get_measures_from_sets(gold_set: set, pred_set: set) -> dict:
    intersection = gold_set & pred_set
    tp = len(intersection)
    fp = len(pred_set - intersection)
    fn = len(gold_set - intersection)
    return calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)


def calc_p_r_f1_from_tp_fp_fn(tp: int, fp: int, fn: int) -> dict:
    p = safe_division(tp, tp + fp)
    r = safe_division(tp, tp + fn)
    f1 = safe_division(2 * p * r, p + r)

    return {"p": p, "r": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def tagging_prf1(gold_ents, pred_ents, type_idx=1) -> dict:
    measure_results = {
        "micro": DEFAULT_PRF1_RESULT_DICT.copy(),
    }
    if type_idx is not None:
        measure_results["macro"] = DEFAULT_PRF1_RESULT_DICT.copy()
        measure_results["types"] = defaultdict(lambda: DEFAULT_PRF1_RESULT_DICT.copy())
    for one_ins_gold_ents, one_ins_pred_ents in zip(gold_ents, pred_ents):
        _result = get_measures_from_sets(set(one_ins_gold_ents), set(one_ins_pred_ents))
        measure_results["micro"]["tp"] += _result["tp"]
        measure_results["micro"]["fp"] += _result["fp"]
        measure_results["micro"]["fn"] += _result["fn"]

        if type_idx is not None:
            _type2ent = defaultdict(lambda: {"gold": set(), "pred": set()})
            for ent in one_ins_gold_ents:
                ent_type = ent[type_idx]
                _type2ent[ent_type]["gold"].add(ent)
            for ent in one_ins_pred_ents:
                ent_type = ent[type_idx]
                _type2ent[ent_type]["pred"].add(ent)
            for ent_type in _type2ent:
                _result = get_measures_from_sets(
                    _type2ent[ent_type]["gold"], _type2ent[ent_type]["pred"]
                )
                measure_results["types"][ent_type]["tp"] += _result["tp"]
                measure_results["types"][ent_type]["fp"] += _result["fp"]
                measure_results["types"][ent_type]["fn"] += _result["fn"]

    # micro
    measure_results["micro"] = calc_p_r_f1_from_tp_fp_fn(
        measure_results["micro"]["tp"],
        measure_results["micro"]["fp"],
        measure_results["micro"]["fn"],
    )

    # for each type
    if type_idx is not None:
        for ent_type in measure_results["types"]:
            measure_results["types"][ent_type] = calc_p_r_f1_from_tp_fp_fn(
                measure_results["types"][ent_type]["tp"],
                measure_results["types"][ent_type]["fp"],
                measure_results["types"][ent_type]["fn"],
            )

        measure_results["types"] = dict(measure_results["types"])

        # macro
        macro_results = defaultdict(list)
        for ent_type in measure_results["types"]:
            for key in measure_results["types"][ent_type]:
                macro_results[key].append(measure_results["types"][ent_type][key])
        for key in macro_results:
            measure_results["macro"][key] = safe_division(
                sum(macro_results[key]), len(macro_results[key])
            )

    return measure_results


def calc_char_event(golds, preds):

    def _match_arg_char_f1(gold_arg, pred_args):
        gtype, grole, gstring = gold_arg
        gchars = set(gstring)
        garg_len = len(gchars)
        cands = []
        for parg in pred_args:
            if parg[0] == gtype and parg[1] == grole:
                pchars = set(str(parg[-1]))
                parg_len = len(pchars)
                pmatch = len(pchars & gchars)
                p = safe_division(pmatch, parg_len)
                r = safe_division(pmatch, garg_len)
                f1 = safe_division(2 * p * r, p + r)
                cands.append(f1)
        if len(cands) > 0:
            f1 = sorted(cands)[-1]
            return f1
        else:
            return 0.0

    pscore = num_gargs = num_pargs = 0
    for _golds, _preds in zip(golds, preds):
        # _golds and _preds pair in one data instance
        gold_args = []
        pred_args = []
        for gold in _golds:
            for arg in gold.get("arguments", []):
                gold_args.append(
                    (gold.get("event_type"), arg.get("role"), arg.get("argument"))
                )
        for pred in _preds:
            for arg in pred.get("arguments", []):
                pred_args.append(
                    (pred.get("event_type"), arg.get("role"), arg.get("argument"))
                )

        num_gargs += len(gold_args)
        num_pargs += len(pred_args)
        for gold_arg in gold_args:
            pscore += _match_arg_char_f1(gold_arg, pred_args)

    p = safe_division(pscore, num_pargs)
    r = safe_division(pscore, num_gargs)
    f1 = safe_division(2 * p * r, p + r)
    return {
        "p": p,
        "r": r,
        "f1": f1,
        "pscore": pscore,
        "num_pargs": num_pargs,
        "num_gargs": num_gargs,
    }


def calc_trigger_identification_metrics(golds, preds):
    tp = fp = fn = 0
    for _golds, _preds in zip(golds, preds):
        gold_triggers = {gold["trigger"] for gold in _golds}
        pred_triggers = {pred["trigger"] for pred in _preds}
        tp += len(gold_triggers & pred_triggers)
        fp += len(pred_triggers - gold_triggers)
        fn += len(gold_triggers - pred_triggers)
    metrics = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    return metrics


def calc_trigger_classification_metrics(golds, preds):
    tp = fp = fn = 0
    for _golds, _preds in zip(golds, preds):
        gold_tgg_cls = {(gold["trigger"], gold["event_type"]) for gold in _golds}
        pred_tgg_cls = {(pred["trigger"], pred["event_type"]) for pred in _preds}
        tp += len(gold_tgg_cls & pred_tgg_cls)
        fp += len(pred_tgg_cls - gold_tgg_cls)
        fn += len(gold_tgg_cls - pred_tgg_cls)
    metrics = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    return metrics


def calc_arg_identification_metrics(golds, preds):
    tp = fp = fn = 0
    for _golds, _preds in zip(golds, preds):
        gold_args = set()
        pred_args = set()
        for gold in _golds:
            _args = {
                (arg["role"], arg["argument"], gold["event_type"])
                for arg in gold["arguments"]
            }
            gold_args.update(_args)
        for pred in _preds:
            _args = {
                (arg["role"], arg["argument"], pred["event_type"])
                for arg in pred["arguments"]
            }
            pred_args.update(_args)
        # logic derived from OneIE
        _tp = 0
        _tp_fp = len(pred_args)
        _tp_fn = len(gold_args)
        _gold_args_wo_role = {_ga[1:] for _ga in gold_args}
        for pred_arg in pred_args:
            if pred_arg[1:] in _gold_args_wo_role:
                _tp += 1
        tp += _tp
        fp += _tp_fp - _tp
        fn += _tp_fn - _tp
    metrics = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    return metrics


def calc_arg_classification_metrics(golds, preds):
    tp = fp = fn = 0
    for _golds, _preds in zip(golds, preds):
        gold_arg_cls = set()
        pred_arg_cls = set()
        for gold in _golds:
            _args = {
                (arg["argument"], arg["role"], gold["event_type"])
                for arg in gold["arguments"]
            }
            gold_arg_cls.update(_args)
        for pred in _preds:
            _args = {
                (arg["argument"], arg["role"], pred["event_type"])
                for arg in pred["arguments"]
            }
            pred_arg_cls.update(_args)
        tp += len(gold_arg_cls & pred_arg_cls)
        fp += len(pred_arg_cls - gold_arg_cls)
        fn += len(gold_arg_cls - pred_arg_cls)
    metrics = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    return metrics


def calc_ent(golds, preds):
    """
    Args:
        golds, preds: [(type, index list), ...]
    """
    res = tagging_prf1(golds, preds, type_idx=0)
    return res


def calc_rel(golds, preds):
    gold_ents = []
    pred_ents = []
    gold_rels = []
    pred_rels = []
    for gold, pred in zip(golds, preds):
        gold_ins_ents = []
        gold_ins_rels = []
        for t in gold:
            gold_ins_ents.extend(t[1:])
            gold_ins_rels.append(t[0])
        gold_ents.append(gold_ins_ents)
        gold_rels.append(gold_ins_rels)
        pred_ins_ents = []
        pred_ins_rels = []
        for t in pred:
            pred_ins_ents.extend(t[1:])
            pred_ins_rels.append(t[0])
        pred_ents.append(pred_ins_ents)
        pred_rels.append(pred_ins_rels)

    metrics = {
        "ent": tagging_prf1(gold_ents, pred_ents, type_idx=None),
        "rel": tagging_prf1(gold_rels, pred_rels, type_idx=None),
    }
    return metrics


def calc_cls(golds, preds):
    metrics = {
        "mcc": -1,
        "acc": -1,
        "mf1": tagging_prf1(golds, preds, type_idx=None),
    }
    y_true = []
    y_pred = []
    for gold, pred in zip(golds, preds):
        y_true.append(" ".join(sorted(gold)))
        y_pred.append(" ".join(sorted(pred)))
    if y_true and y_pred:
        metrics["acc"] = accuracy_score(y_true, y_pred)
    else:
        metrics["acc"] = 0.0
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    return metrics


def calc_span(golds, preds, mode="span"):
    def _get_tokens(spans: list[tuple[tuple[int]]]) -> list[int]:
        tokens = []
        for span in spans:
            for part in span:
                _toks = []
                if len(part) == 1:
                    _toks = [part[0]]
                elif len(part) > 1:
                    if mode == "w2":
                        _toks = [*part]
                    elif mode == "span":
                        _toks = [*range(part[0], part[1] + 1)]
                    else:
                        raise ValueError
                tokens.extend(_toks)
        return tokens

    metrics = {
        "em": -1,
        "f1": None,
    }
    acc_num = 0
    tp = fp = fn = 0
    for gold, pred in zip(golds, preds):
        if gold == pred:
            acc_num += 1
        gold_tokens = _get_tokens(gold)
        pred_tokens = _get_tokens(pred)
        tp += len(set(gold_tokens) & set(pred_tokens))
        fp += len(set(pred_tokens) - set(gold_tokens))
        fn += len(set(gold_tokens) - set(pred_tokens))
    if len(golds) > 0:
        metrics["em"] = acc_num / len(golds)
    else:
        metrics["em"] = 0.0
    metrics["f1"] = calc_p_r_f1_from_tp_fp_fn(tp, fp, fn)
    return metrics


def rel_eval(self, pred_result, use_name=False):
    correct = 0  # tp + tn
    total = len(self.data)  # tp + tn + fp + fn
    correct_positive = 0  # tp
    pred_positive = 0  # tp + fp
    gold_positive = 0  # tp + fn
    correct_category = np.zeros([31, 1])  # correct nums in each category
    pre_category = np.zeros([31, 1])  # real nums in each category
    n_category = np.zeros([31, 1])  # predict nums in each category
    data_with_pred_T = []
    data_with_pred_F = []
    neg = -1
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for name in ['NA', 'na', 'no_relation', 'Other', 'Others', 'none', 'None']:
        if name in self.rel2id:
            if use_name:
                neg = name
            else:
                neg = self.rel2id[name]
            break
    y_pred = []
    y_gt = []
    for i in range(total):
        y_pred.append(pred_result[i])
        y_gt.append(self.rel2id[self.data[i]['relation']])
        if use_name:
            golden = self.data[i]['relation']
        else:
            golden = self.rel2id[self.data[i]['relation']]  # Ground Truth Label
            n_category[golden] += 1
        data_with_pred = (str(self.data[i]) + str(pred_result[i]))
        if golden == pred_result[i]:
            data_with_pred_T.append(data_with_pred)
            if golden != neg:
                tp += 1
                correct_category[golden] += 1
            else:
                tn += 1
                correct_category[0] += 1
        else:
            if pred_result[i] != neg:
                fp += 1
                pre_category[pred_result[i]] += 1
            else:
                fn += 1
                pre_category[0] += 1

            data_with_pred_F.append(data_with_pred)

    acc = float(tp + tn) / float(total)
    try:
        micro_p = float(tp) / float(tp + fp)
    except:
        micro_p = 0
    try:
        micro_r = float(tp) / float(tp + fn)
    except:
        micro_r = 0
    try:
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)
    except:
        micro_f1 = 0

    result = {'acc': acc, 'micro_p': micro_p, 'micro_r': micro_r, 'micro_f1': micro_f1}
    # logging.info('Evaluation result: {}.'.format(result))
    return result, correct_category, pre_category, n_category, data_with_pred_T, data_with_pred_F


class MultiPartSpanMetric:
    def _encode_span_to_label_dict(self, span_to_label: dict) -> list:
        span_to_label_list = []
        for key, val in span_to_label.items():
            span_to_label_list.append({"key": key, "val": val})
        return span_to_label_list

    def _decode_span_to_label(self, span_to_label_list: list) -> dict:
        span_to_label = {}
        for content in span_to_label_list:
            span_to_label[tuple(content["key"])] = content["val"]
        return span_to_label

    def get_instances_from_batch(self, raw_batch: dict, out_batch: dict) -> Tuple:
        gold_instances = []
        pred_instances = []

        batch_gold = []
        batch_size = len(next(iter(raw_batch.values())))  # 获取每个键对应的样本数量
        for i in range(batch_size):
            sample = {key: raw_batch[key][i] for key in raw_batch}
            batch_gold.append(sample)

        assert len(batch_gold) == len(out_batch["pred"])

        for i, gold in enumerate(batch_gold):
            ins_id = gold["raw"].get("id")
            # encode to list to make the span_to_label dict json-serializable
            # where the original dict key is a tuple
            span_to_label_list = self._encode_span_to_label_dict(gold["span_to_label"])
            gold["span_to_label"] = span_to_label_list
            # print(gold["spans"])
            gold_instances.append(
                {
                    "id": ins_id,
                    "span_to_label_list": span_to_label_list,
                    "raw_gold_content": gold,
                    "spans": [
                        tuple(
                            tuple(sub_span) if isinstance(sub_span, list) else sub_span for sub_span in multi_part_span)
                        for
                        multi_part_span in gold["spans"]],
                }
            )
            pred_instances.append(
                {
                    "id": ins_id,
                    "spans":[
                        tuple(
                            tuple(sub_span) if isinstance(sub_span, list) else sub_span for sub_span in multi_part_span)
                        for
                        multi_part_span in out_batch["pred"][i]],
                }
            )

        return gold_instances, pred_instances

    def calculate_scores(self, golds: list, preds: list) -> dict:
        # for general purpose evaluation
        general_gold_spans, general_pred_spans = [], []
        # cls task
        gold_cls_list, pred_cls_list = [], []
        # ent task
        gold_ent_list, pred_ent_list = [], []
        gold_ent_list_15, pred_ent_list_15 = [], []
        gold_ent_list_17, pred_ent_list_17 = [], []
        # rel task
        gold_rel_list, pred_rel_list = [], []
        # event task
        gold_event_list, pred_event_list = [], []
        # span task
        gold_span_list, pred_span_list = [], []
        # discon ent task
        gold_discon_ent_list, pred_discon_ent_list = [], []
        # hyper rel task
        gold_hyper_rel_list, pred_hyper_rel_list = [], []

        for gold, pred in zip(golds, preds):
            general_gold_spans.append(gold["spans"])
            general_pred_spans.append(pred["spans"])
            span_to_label = self._decode_span_to_label(gold["span_to_label_list"])
            gold_clses, pred_clses = [], []
            gold_ents, pred_ents = [], []
            gold_rels, pred_rels = [], []
            gold_trigger_to_event = defaultdict(
                lambda: {"event_type": "", "arguments": []}
            )
            pred_trigger_to_event = defaultdict(
                lambda: {"event_type": "", "arguments": []}
            )
            gold_events, pred_events = [], []
            gold_spans, pred_spans = [], []
            gold_ents_15, pred_ents_15 = [], []
            gold_ents_17, pred_ents_17 = [], []
            gold_discon_ents, pred_discon_ents = [], []
            gold_hyper_rels, pred_hyper_rels = [], []

            raw_schema = gold["raw_gold_content"]["raw"]["schema"]
            for span in gold["spans"]:
                #print(span_to_label)
                if span[0] in span_to_label:
                    label = span_to_label[span[0]]
                    #print(label)
                    if label["task"] == "cls" and len(span) == 1:
                        gold_clses.append(label["string"])
                    elif label["task"] == "ent" and len(span) == 2:
                        gold_ents.append((label["string"], *span[1:]))
                        if "15" in gold["id"]:
                            gold_ents_15.append((label["string"], *span[1:]))
                        elif "17" in gold["id"]:
                            gold_ents_17.append((label["string"], *span[1:]))
                    elif label["task"] == "rel" and len(span) == 3:
                        gold_rels.append((label["string"], *span[1:]))
                    elif label["task"] == "event":
                        if label["type"] == "lm" and len(span) == 2:
                            gold_trigger_to_event[span[1]]["event_type"] = label["string"]  # fmt: skip
                        elif label["type"] == "lr" and len(span) == 3:
                            gold_trigger_to_event[span[1]]["arguments"].append(
                                {"argument": span[2], "role": label["string"]}
                            )
                    elif label["task"] == "discontinuous_ent" and len(span) > 1:
                        gold_discon_ents.append((label["string"], *span[1:]))
                    elif label["task"] == "hyper_rel" and len(span) == 5 and span[3] in span_to_label:  # fmt: skip
                        q_label = span_to_label[span[3]]
                        gold_hyper_rels.append(
                            (label["string"], span[1], span[2], q_label["string"], span[4]))  # fmt: skip
                else:
                    # span task has no labels
                    gold_spans.append(tuple(span))
            for trigger, item in gold_trigger_to_event.items():
                legal_roles = raw_schema["event"][item["event_type"]]
                gold_events.append(
                    {
                        "trigger": trigger,
                        "event_type": item["event_type"],
                        "arguments": [
                            arg
                            for arg in filter(
                                lambda arg: arg["role"] in legal_roles,
                                item["arguments"],
                            )
                        ],
                    }
                )

            for span in pred["spans"]:
                if span[0] in span_to_label:
                    label = span_to_label[span[0]]
                    if label["task"] == "cls" and len(span) == 1:
                        pred_clses.append(label["string"])
                    elif label["task"] == "ent" and len(span) == 2:
                        pred_ents.append((label["string"], *span[1:]))
                        if "15" in pred["id"]:
                            pred_ents_15.append((label["string"], *span[1:]))
                        elif "17" in pred["id"]:
                            pred_ents_17.append((label["string"], *span[1:]))
                    elif label["task"] == "rel" and len(span) == 3:
                        pred_rels.append((label["string"], *span[1:]))
                    elif label["task"] == "event":
                        if label["type"] == "lm" and len(span) == 2:
                            pred_trigger_to_event[span[1]]["event_type"] = label["string"]  # fmt: skip
                        elif label["type"] == "lr" and len(span) == 3:
                            pred_trigger_to_event[span[1]]["arguments"].append(
                                {"argument": span[2], "role": label["string"]}
                            )
                    elif label["task"] == "discontinuous_ent" and len(span) > 1:
                        pred_discon_ents.append((label["string"], *span[1:]))
                    elif label["task"] == "hyper_rel" and len(span) == 5 and span[3] in span_to_label:  # fmt: skip
                        q_label = span_to_label[span[3]]
                        pred_hyper_rels.append(
                            (label["string"], span[1], span[2], q_label["string"], span[4]))  # fmt: skip
                else:
                    # span task has no labels
                    pred_spans.append(tuple(span))
            for trigger, item in pred_trigger_to_event.items():
                if item["event_type"] not in raw_schema["event"]:
                    continue
                legal_roles = raw_schema["event"][item["event_type"]]
                pred_events.append(
                    {
                        "trigger": trigger,
                        "event_type": item["event_type"],
                        "arguments": [
                            arg
                            for arg in filter(
                                lambda arg: arg["role"] in legal_roles,
                                item["arguments"],
                            )
                        ],
                    }
                )

            gold_cls_list.append(gold_clses)
            pred_cls_list.append(pred_clses)
            gold_ent_list.append(gold_ents)
            pred_ent_list.append(pred_ents)
            if len(gold_ents_15):
                gold_ent_list_15.append(gold_ents_15)
            if len(gold_ents_17):
                gold_ent_list_17.append(gold_ents_17)

            pred_ent_list_15.append(pred_ents_15)

            pred_ent_list_17.append(pred_ents_17)
            gold_rel_list.append(gold_rels)
            pred_rel_list.append(pred_rels)
            gold_event_list.append(gold_events)
            pred_event_list.append(pred_events)
            gold_span_list.append(gold_spans)
            pred_span_list.append(pred_spans)
            gold_discon_ent_list.append(gold_discon_ents)
            pred_discon_ent_list.append(pred_discon_ents)
            gold_hyper_rel_list.append(gold_hyper_rels)
            pred_hyper_rel_list.append(pred_hyper_rels)
        # assert len(gold_ent_list_15)+len(gold_ent_list_17)==len(gold_ent_list)
        # assert len(pred_ent_list_15) + len(pred_ent_list_17) == len(pred_ent_list)

        metrics = {
            "general_spans": tagging_prf1(
                general_gold_spans, general_pred_spans, type_idx=None
            ),
            "cls": calc_cls(gold_cls_list, pred_cls_list),
            "ent": calc_ent(gold_ent_list, pred_ent_list),
            # "twitter15": calc_ent(gold_ent_list_15, pred_ent_list_15),
            # "twitter17": calc_ent(gold_ent_list_17, pred_ent_list_17),
            "rel": calc_rel(gold_rel_list, pred_rel_list),
            "event": {
                "trigger_id": calc_trigger_identification_metrics(
                    gold_event_list, pred_event_list
                ),
                "trigger_cls": calc_trigger_classification_metrics(
                    gold_event_list, pred_event_list
                ),
                "arg_id": calc_arg_identification_metrics(
                    gold_event_list, pred_event_list
                ),
                "arg_cls": calc_arg_classification_metrics(
                    gold_event_list, pred_event_list
                ),
                "char_event": calc_char_event(gold_event_list, pred_event_list),
            },
            # "span": tagging_prf1(gold_span_list, pred_span_list, type_idx=None),
            "span": calc_span(gold_span_list, pred_span_list),
        }

        return metrics

    def add_batch(self, raw_batch: dict, out_batch: dict) -> dict:
        gold_instances, pred_instances = self.get_instances_from_batch(
            raw_batch, out_batch
        )

        batch_result = {
            "gold": gold_instances,
            "pred": pred_instances,
            "metric_scores": self.calculate_scores(gold_instances, pred_instances),
        }

        return batch_result
