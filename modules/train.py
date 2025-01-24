# -*- coding: utf-8 -*-
import re
import os
from collections import defaultdict

import torch
from torch import optim
from tqdm import tqdm
import random
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
from modules.metric import MultiPartSpanMetric
from torch import nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PrefixTuningConfig, TaskType, PeftType

class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

class Trainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None,
                 args=None, logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.args = args
        self.model = model
        # if self.args.use_ldnet:
            
        #     self.freeze_mulmirror()
        # else:
        #     if self.args.use_llm or self.args.use_qwen:
        #         self.freeze_pretrained_model()
        #         self.load_lora_model(self.model)
        # self.model = torch.nn.DataParallel(self.model, device_ids=[1,2])
        self.processor = processor
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_train_metric = 99999999
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.best_train_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.metric=MultiPartSpanMetric()

    def find_all_linear_names(self, model):
      
        cls = nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        lora_module_names = list(lora_module_names)
        print(f'LoRA target module names: {lora_module_names}')
        return lora_module_names
    
    def load_lora_model(self, model):
        # target_modules = self.find_all_linear_names(model)
        target_modules = ['linear_q_fine', 'linear_k_fine', 'linear_v_fine', 'linear_q_coarse', 'linear_k_coarse', 'linear_v_coarse', 'combine_linear', 'combine_linear1', 'cls_layer.proj', 'pic_mlp.0', 'pic_mlp.3', 'pic_mlp.6']

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
        # print(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
        # model.print_trainable_parameters()
        return model
            
    def freeze_pretrained_model(self):
        for name, param in self.model.named_parameters():
            if 'plm' in name or 'image_model' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True   

    def freeze_mulmirror(self):
        trainable_params = ['pointer.gp.dense.weight', 'pointer.gp.dense.bias', 'bce_linear.weight', 'bce_linear.bias']
        for name, param in self.model.named_parameters():
            if name not in trainable_params:
                param.requires_grad = False
            else:
                param.requires_grad = True    

    def train(self):
        self.multiModal_before_train()

        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load('./ckpt/'+self.args.load_path+'/best_model.pth'), strict=False)
            # if os.path.exists('./ckpt/'+self.args.load_path+'/optimizer.pth'):
            #     self.optimizer.load_state_dict(torch.load('./ckpt/'+self.args.load_path+'/optimizer.pth'))
            self.logger.info("Load vit model successful!")
        if self.args.load_ldnet_path is not None:
            l = torch.load(self.args.load_ldnet_path)
            self.model.load_state_dict(l['model_state'], strict=False)
            # self.optimizer.load_state_dict(l["optimizer_state"])
            self.logger.info("Load ldnet model successful!")

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0

            for epoch in range(1, self.args.num_epochs + 1):
                total_loss = 0
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = {key: value.to(self.args.device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
                    # print(batch)
                    output = self.model(**batch)

                    loss=output["loss"].mean()
                    if self.args.use_ldnet:
                        bce_loss=output["bce_loss"]

                    avg_loss += loss.detach().cpu().item()
                    # loss.requires_grad_(True)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        total_loss += avg_loss
                        avg_loss = float(avg_loss) / self.refresh_step
                        if self.args.use_ldnet:
                            print_output = "loss:{:<6.5f}, bce_loss:{:<6.5f}".format(avg_loss, bce_loss)
                        else:
                            print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss,
                                                   global_step=self.step)  # tensorbordx
                        avg_loss = 0

                self.logger.info("***** Train Eval results *****")
                self.logger.info("\n%s", total_loss)

                if self.writer:
                    self.writer.add_scalar(tag='train_loss', scalar_value=total_loss, global_step=epoch)  # tensorbordx
                self.logger.info("Epoch {}/{}, best train loss: {}, best epoch: {}, current train loss: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_train_metric, self.best_train_epoch,
                                         total_loss))
                if total_loss < self.best_train_metric:
                    self.best_train_metric = total_loss
                    self.best_train_epoch = epoch

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)  # generator to dev.

            torch.cuda.empty_cache()

            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev loss is {}".format(self.best_dev_epoch,
                                                                                                    self.best_dev_metric))
            self.logger.info(
                "Get best test performance at epoch {}, best test loss is {}".format(self.best_test_epoch,
                                                                                         self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        y_true, y_pred = [], []
        step = 0

        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1

                    batch = {key: value.to(self.args.device) if isinstance(value, torch.Tensor) else value for
                             key, value in batch.items()}

                    output = self.model(**batch,is_eval=True)
                    loss = output["loss"].mean()

                    total_loss += loss.detach().cpu().item()

                    batch_results: dict = self.metric.add_batch(batch, output)

                    for gold, pred in zip(batch_results["gold"], batch_results["pred"]):
                        y_true.append(gold)
                        y_pred.append(pred)

                    pbar.update()
                pbar.close()
                results = self.metric.calculate_scores(y_true, y_pred)
                self.logger.info("***** Dev Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results["general_spans"]["micro"]["f1"])
                if self.writer:
                    self.writer.add_scalar(tag='dev_f1', scalar_value=f1_score, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss / step,
                                           global_step=epoch)  # tensorbordx

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                                         f1_score))
                if f1_score >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = f1_score  # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), './ckpt/' + self.args.save_path + "/best_model.pth")
                        torch.save(self.optimizer.state_dict(), './ckpt/' + self.args.save_path + "/optimizer.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self):
        self.model.to('cuda')
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        self.logger.info("Loading model from {}".format(self.args.save_path))
        self.model.load_state_dict(torch.load('./ckpt/'+self.args.save_path+'/best_model.pth'), strict=False)
        # if os.path.exists('./ckpt/'+self.args.save_path+'/optimizer.pth'):
        #     self.optimizer.load_state_dict(torch.load('./ckpt/'+self.args.save_path+'/optimizer.pth'))
        self.logger.info("Load vit model successful!")
        # if self.args.load_ldnet_path is not None:
        #     l = torch.load(self.args.load_ldnet_path)
        #     self.model.load_state_dict(l['model_state'], strict=False)
        #     # self.optimizer.load_state_dict(l["optimizer_state"])
        #     self.logger.info("Load ldnet model successful!")
        y_true, y_pred = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = {key: value.to(self.args.device) if isinstance(value, torch.Tensor) else value for
                             key, value in batch.items()}
                    output = self.model(**batch, is_eval=True)
                    loss = output["loss"].mean()

                    total_loss += loss.detach().cpu().item()

                    batch_results: dict = self.metric.add_batch(batch, output)

                    for gold, pred in zip(batch_results["gold"], batch_results["pred"]):
                        y_true.append(gold)
                        y_pred.append(pred)
                    print_output = "spans.f1:{:<6.5f}".format(batch_results['metric_scores']["general_spans"]["micro"]["f1"])
                    pbar.set_postfix_str(print_output)
                    pbar.update()
                # evaluate done
                pbar.close()


                results = self.metric.calculate_scores(y_true, y_pred)


                self.logger.info("***** Test Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results["general_spans"]["micro"]["f1"])
                if self.writer:
                    self.writer.add_scalar(tag='test_f1', scalar_value=f1_score)  # tensorbordx
                    self.writer.add_scalar(tag='test_loss',
                                           scalar_value=total_loss / len(self.test_data))  # tensorbordx
                total_loss = 0
                self.logger.info("Test f1 score: {}.".format(f1_score))

        self.model.train()

    def bert_before_train(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=0.01)

        self.model.to(self.args.device)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)

    def multiModal_before_train(self):
        no_decay = r"(embedding|LayerNorm|\.bias$)"
        plm_lr = r"^plm\."
        # non_trainable = r"^plm\.(emb|encoder\.layer\.[0-3])"
        non_trainable = "no_non_trainable"

        param_groups = []
        for name, param in self.model.named_parameters():
            if 'encoder_conv' in name or 'gates' in name:
                param_groups.append(
                    {"params": param, "lr": 3e-5, "weight_decay": 1e-2}
                )
            # elif 'plm' in name and "embed_tokens" not in name:
            #     continue
            else:
                # print(name)
                lr = self.args.lr
                weight_decay = self.args.weight_decay
                if re.search(non_trainable, name):
                    param.requires_grad = False
                if not re.search(plm_lr, name):
                    lr = self.args.other_lr
                if re.search(no_decay, name):
                    weight_decay = 0.0
                param_groups.append(
                    {"params": param, "lr": lr, "weight_decay": weight_decay}
                )

        # for name, par in self.model.named_parameters():  # freeze resnet
        #     if 'image_model' in name or "plm" in name:
        #         par.requires_grad = False

        if self.args.use_llm:
            for name, par in self.model.named_parameters():  # freeze qwen decoder
                if 'plm' in name and "embed_tokens" not in name:
                    par.requires_grad = False

        self.optimizer = optim.AdamW(param_groups,
            lr=self.args.lr,
            betas=(0.9, 0.98),
            eps=1e-6,)

        for name, par in self.model.named_parameters():  # freeze resnet
            if 'image_model' in name:   par.requires_grad = False

        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)
