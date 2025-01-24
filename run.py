import os
import argparse
import logging
import sys
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
sys.path.append("..")

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from models.model import SchemaGuidedInstructBertModel
from processor.dataset import Mirrorprocessor,MulMirrorDataset,collate_fn
from modules.train import Trainer


import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


DATA_PATH = {
    'twitter15': {
                # input text data
                'train': './data/twitter2015/train.txt',
                'dev':  './data/twitter2015/val.txt',
                'test':  './data/twitter2015/test.txt',

                # visual objects data
                'train_auximgs':  './data/twitter2015/twitter2015_train_dict.pth',
                'dev_auximgs':  './data/twitter2015/twitter2015_val_dict.pth',
                'test_auximgs':  './data/twitter2015/twitter2015_test_dict.pth',
                'train_auximgs_dif':  './data/twitter2015/train_grounding_cut_dif.pth',
                'dev_auximgs_dif':  './data/twitter2015/val_grounding_cut_dif.pth',
                'test_auximgs_dif':  './data/twitter2015/test_grounding_cut_dif.pth',

                # correlation coefficient data
                'train_weak_ori': './data/twitter2015/ner_train_weight_weak.txt',
                'dev_weak_ori': './data/twitter2015/ner_val_weight_weak.txt',
                'test_weak_ori': './data/twitter2015/ner_test_weight_weak.txt',
                'train_strong_ori': './data/twitter2015/ner_train_weight_strong.txt',
                'dev_strong_ori': './data/twitter2015/ner_val_weight_strong.txt',
                'test_strong_ori': './data/twitter2015/ner_test_weight_strong.txt',
                'train_weak_dif': './data/twitter2015/ner_diff_train_weight_weak.txt',
                'dev_weak_dif': './data/twitter2015/ner_diff_val_weight_weak.txt',
                'test_weak_dif': './data/twitter2015/ner_diff_test_weight_weak.txt',
                'train_strong_dif': './data/twitter2015/ner_diff_train_weight_strong.txt',
                'dev_strong_dif': './data/twitter2015/ner_diff_val_weight_strong.txt',
                'test_strong_dif': './data/twitter2015/ner_diff_test_weight_strong.txt',
                
                # phrase text data
                'train_grounding_text':'./data/twitter2015/ner15_grounding_text_train.json',
                'dev_grounding_text': './data/twitter2015/ner15_grounding_text_val.json',
                'test_grounding_text': './data/twitter2015/ner15_grounding_text_test.json',
    },

    'twitter17': {
                # text data
                'train': './data/twitter2017/train.txt',
                'dev':  './data/twitter2017/valid.txt',
                'test':  './data/twitter2017/test.txt',
        
                # visual objects data
                'train_auximgs': './data/twitter2017/twitter2017_train_dict.pth',
                'dev_auximgs': './data/twitter2017/twitter2017_val_dict.pth',
                'test_auximgs': './data/twitter2017/twitter2017_test_dict.pth',
                'train_auximgs_dif':  './data/twitter2017/train_grounding_cut_dif.pth',
                'dev_auximgs_dif':  './data/twitter2017/val_grounding_cut_dif.pth',
                'test_auximgs_dif':  './data/twitter2017/test_grounding_cut_dif.pth',

                # correlation coefficient data
                'train_weak_ori': './data/twitter2017/ner_train_weight_weak.txt',
                'dev_weak_ori': './data/twitter2017/ner_val_weight_weak.txt',
                'test_weak_ori': './data/twitter2017/ner_test_weight_weak.txt',
                'train_strong_ori': './data/twitter2017/ner_train_weight_strong.txt',
                'dev_strong_ori': './data/twitter2017/ner_val_weight_strong.txt',
                'test_strong_ori': './data/twitter2017/ner_test_weight_strong.txt',
                'train_weak_dif': './data/twitter2017/diff_ner_train_weight_weak.txt',
                'dev_weak_dif': './data/twitter2017/diff_ner_val_weight_weak.txt',
                'test_weak_dif': './data/twitter2017/diff_ner_test_weight_weak.txt',
                'train_strong_dif': './data/twitter2017/diff_ner_train_weight_strong.txt',
                'dev_strong_dif': './data/twitter2017/diff_ner_val_weight_strong.txt',
                'test_strong_dif': './data/twitter2017/diff_ner_test_weight_strong.txt',

                # phrase text data
                'train_grounding_text': './data/twitter2017/ner17_grounding_text_train.json',
                'dev_grounding_text': './data/twitter2017/ner17_grounding_text_val.json',
                'test_grounding_text': './data/twitter2017/ner17_grounding_text_test.json',
            },
        
}

# original image data
IMG_PATH = {
    'twitter15': './data/twitter2015_images',
    'twitter17': './data/twitter2017_images',
}

# generated image data
IMG_PATH_dif = {
    'twitter15': {
        'train': './data/ner15_diffusion_pic/train/',
        'test': './data/ner15_diffusion_pic/test/',
        'dev':  './data/ner15_diffusion_pic/val/',
    },
    'twitter17': {
        'train': './data/ner17_diffusion_pic',
        'test': './data/ner17_diffusion_pic',
        'dev': './data/ner17_diffusion_pic',
    }
}

# auxiliary images for visual objects
AUX_PATH = {
    'twitter15': {
                'train': './data/twitter2015_aux_images/train/crops',
                'dev': './data/twitter2015_aux_images/val/crops',
                'test': './data/twitter2015_aux_images/test/crops',
            },

    'twitter17': {
                'train': './data/twitter2017_aux_images/train/crops',
                'dev': './data/twitter2017_aux_images/val/crops',
                'test': './data/twitter2017_aux_images/test/crops',
            }
}


def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/newInst/RE/data/mre/', type=str,help="The name of dataset.")
    parser.add_argument('--encoder_name', default='deberta-v3-large', type=str, help="Pretrained language model path")

    parser.add_argument('--bce_mean', default=False, type=bool, help="whether to use mean loss")
    parser.add_argument('--use_images', default=False, type=bool,help="whether to use mean loss")
    parser.add_argument('--use_ldnet', default=False, type=bool,help="whether to use mean loss")
    parser.add_argument('--use_llm', default=False, type=bool,help="whether to use mean loss")
    parser.add_argument('--use_qwen', default=False, type=bool,help="whether to use mean loss")
    parser.add_argument('--do_train', default=False, type=bool)
    parser.add_argument('--use_inst', default=False, type=bool)
    parser.add_argument('--droprate', default=False, type=bool)
    parser.add_argument('--use_only_mr', default=False, type=bool)
    parser.add_argument('--use_ldnet_ablation', default=False, type=bool)

    parser.add_argument('--num_epochs', default=6, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda:0', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--lr', default=3e-5, type=float, help="learning rate")
    parser.add_argument('--other_lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('--weight_decay', default=0.1, type=float, help="weight_delay")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=20, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=2021, type=int, help="random seed, default is 1")
    parser.add_argument('--load_ldnet_path', default=None, type=str)
    parser.add_argument('--load_path', default=None, type=str,help="Load model from load_path")
    parser.add_argument('--save_path', default="deberta-v3-large-large_mlp_entity_17", type=str, help="save model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
  
    parser.add_argument('--max_seq', default=256, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")

    args = parser.parse_args()
    # print(args)
    # exit()
    data_path = args.data_path
    model=SchemaGuidedInstructBertModel
    data_process, dataset_class = Mirrorprocessor,MulMirrorDataset

    set_seed(args.seed) # set seed, default is 1
    if args.save_path is not None:  # make save_path dir
        if not os.path.exists('./ckpt/' + args.save_path):
            os.makedirs('./ckpt/' + args.save_path, exist_ok=True)
    print(args)

    logdir = "logs/" + args.data_path.split("/")[1]+"_"+ args.encoder_name.split("/")[-1]+ "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes
    # writer = SummaryWriter(logdir=logdir)
    #writer=None
    writer = SummaryWriter("logs" + "/" + "writer"+"/"+args.data_path.split("/")[1]+ "_"+str(args.batch_size) + "_" + str(args.lr) + args.notes)

    fh = logging.FileHandler(logdir)
    fh.setLevel(logging.DEBUG)
    formatter=logging.Formatter('%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    processor = data_process(data_path, args.encoder_name, args.use_llm, args.use_qwen)
    train_dataset = dataset_class(processor, args.use_llm, args.use_qwen, max_seq=args.max_seq, sample_ratio=args.sample_ratio, mode='train', include_instructions=args.use_inst)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True,collate_fn=collate_fn)

    dev_dataset = dataset_class(processor, max_seq=args.max_seq, sample_ratio=args.sample_ratio, mode='dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,collate_fn=collate_fn)

    test_dataset = dataset_class(processor, max_seq=args.max_seq, sample_ratio=args.sample_ratio, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True,collate_fn=collate_fn)

    model = model(plm_dir=args.encoder_name, vocab_size=len(test_dataset.tokenizer), use_ldnet=args.use_ldnet, use_ldnet_ablation=args.use_ldnet_ablation, use_only_mr=args.use_only_mr, droprate=args.droprate, bce_mean=args.bce_mean,use_images=args.use_images)

    trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                      args=args, logger=logger, writer=writer)
    if args.do_train:
        # train
        trainer.train()
        trainer.test()

    else:
        # only do test
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()

if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1 python run.py --data_path='data/newInst/NER/data/twitter2015/' --bce_mean=True --num_epochs=6 --eval_begin_epoch=6 --batch_size=32 --save_path=deberta-v3-large-large_mlp_entity_15 --use_images=False --use_llm=False --use_ldnet=False  --load_ldnet_path=/data/share2/strawberry/LDNet/pretrain_outputs/Pretrain_ld_mean/ckpt/SchemaGuidedInstructBertModel.step.2499.pth