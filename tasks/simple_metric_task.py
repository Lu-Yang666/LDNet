import json
import os
from typing import Tuple

import torch

from rex.metrics.base import MetricBase
from rex.tasks.simple_task import SimpleTask
from rex.utils.dict import get_dict_content
from rex.utils.io import dump_json, dump_jsonlines
from rex.utils.logging import logger
from rex.utils.progress_bar import pbar
from pathlib import Path


class SimpleMetricTask(SimpleTask):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.logits_path = Path(config.task_dir).joinpath("logits")
        os.makedirs(self.logits_path, exist_ok=True)

    def initialize(self):
        super().initialize()
        logger.debug("Init metric")
        self.metric = self.init_metric()

    def init_metric(self) -> MetricBase:
        raise NotImplementedError

    @torch.no_grad()
    def eval(
        self, dataset_name, verbose=False, dump=False, dump_middle=True, postfix=""
    ) -> Tuple[float, dict]:
        """Eval on specific dataset and return loss and measurements

        Args:
            dataset_name: which dataset to evaluate
            verbose: whether to log evaluation results
            dump: if True, dump metric results to `self.measures_path`
            dump_middle: if True, dump middle results to `self.middle_path`
            postfix: filepath postfix for dumping

        Returns:
            eval_loss: float
            metrics: dict
        """
        self.model.eval()
        eval_loader = self.get_data_loader(
            dataset_name, is_eval=True, epoch=self.history["curr_epoch"]
        )
        if self.config.generate_logits == True: 
            eval_loader = self.get_data_loader("train", False, self.history["curr_epoch"])
        loader = pbar(eval_loader, desc=f"{dataset_name} - {postfix} Eval", ascii=True)
        eval_loss = 0.0
        tot_batch_results = []

        if self.config.generate_logits == True: logits = []
        label_drop_accuracy = []
        for batch in loader:
            if not self.config.fewshot and not self.config.zeroshot: 
                out = self.model(**batch, is_eval=True)
                if self.config.generate_logits == True: 
                    for i in range(batch['input_ids'].shape[0]):
                        # logits.append({batch['input_ids'][i]:out['logits'][i]})
                        logits.append({batch['raw'][0]['id']:out['logits'][i]})
                eval_loss += out["loss"].item()
                batch_results: dict = self.metric(batch, out)
                batch_metric_score = get_dict_content(
                    batch_results["metric_scores"], self.config.best_metric_field
                )
                loader.set_postfix({self.config.best_metric_field: batch_metric_score})
                batch_instances = [
                    {"gold": gold, "pred": pred}
                    for gold, pred in zip(batch_results["gold"], batch_results["pred"])
                ]
                tot_batch_results.extend(batch_instances)
                label_drop_accuracy.append(out['label_drop_accuracy'].item())
            elif self.config.fewshot and self.history["curr_epoch"] > 5:
                out = self.model(**batch, is_eval=True)
                if self.config.generate_logits == True: 
                    for i in range(batch['input_ids'].shape[0]):
                        # logits.append({batch['input_ids'][i]:out['logits'][i]})
                        logits.append({batch['raw'][0]['id']:out['logits'][i]})
                eval_loss += out["loss"].item()
                batch_results: dict = self.metric(batch, out)
                batch_metric_score = get_dict_content(
                    batch_results["metric_scores"], self.config.best_metric_field
                )
                loader.set_postfix({self.config.best_metric_field: batch_metric_score})
                batch_instances = [
                    {"gold": gold, "pred": pred}
                    for gold, pred in zip(batch_results["gold"], batch_results["pred"])
                ]
                tot_batch_results.extend(batch_instances)
                label_drop_accuracy.append(out['label_drop_accuracy'].item())
            elif self.config.zeroshot and self.history["curr_epoch"] >= 0:

                out = self.model(**batch, is_eval=True)
                if self.config.generate_logits == True: 
                    for i in range(batch['input_ids'].shape[0]):
                        logits.append({batch['raw'][0]['id']:out['logits'][i]})
                eval_loss += out["loss"].item()
                batch_results: dict = self.metric(batch, out)
                batch_metric_score = get_dict_content(
                    batch_results["metric_scores"], self.config.best_metric_field
                )
                loader.set_postfix({self.config.best_metric_field: batch_metric_score})
                batch_instances = [
                    {"gold": gold, "pred": pred}
                    for gold, pred in zip(batch_results["gold"], batch_results["pred"])
                ]
                tot_batch_results.extend(batch_instances)
                label_drop_accuracy.append(out['label_drop_accuracy'].item())
            else:
                pass

        if self.config.generate_logits == True: 
            torch.save(logits, self.config.dump_cache_dir+'/logits.pt')

        # ## label drop accuracy ##
        # if len(label_drop_accuracy) != 0:
        #     average_label_drop_accuracy = sum(label_drop_accuracy) / len(label_drop_accuracy)
        #     logger.info(f"======================average_label_drop_accuracy={average_label_drop_accuracy}=============")
        logger.info(loader)
        measurements = self.metric.compute()

        if verbose:
            logger.info(f"Eval dataset: {dataset_name}")
            logger.info(f"Eval loss: {eval_loss}")
            logger.info(
                f"Eval metrics: {get_dict_content(measurements, self.config.best_metric_field)}"
            )
        _filename_prefix = (
            f"{dataset_name}.{postfix}" if len(postfix) > 0 else f"{dataset_name}"
        )
        if dump:
            dump_obj = {
                "dataset_name": dataset_name,
                "eval_loss": eval_loss,
                "metrics": measurements,
            }
            _measure_result_filepath = self.measures_path.joinpath(
                f"{_filename_prefix}.json"
            )
            dump_json(dump_obj, _measure_result_filepath)
            logger.info(f"Dump measure results into {_measure_result_filepath}")
        # if dump_middle:
        #     _middle_result_filepath = self.middle_path.joinpath(
        #         f"{_filename_prefix}.jsonl"
        #     )
        #     dump_jsonlines(tot_batch_results, _middle_result_filepath)
        #     logger.info(f"Dump middle results into {_middle_result_filepath}")

        self.metric.reset()

        return eval_loss, measurements
