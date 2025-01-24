# LDNet

- Code for ``Label Drop for Multi-Aspect Relation Modeling in Universal Information Extraction``

## Requirements
- Environment
  - Python >= 3.10
  - CUDA 11.8
  - Download deberta-v3-base model from [Huggingface](https://huggingface.co/microsoft/deberta-v3-base) and put it under file ``microsoft/deberta-v3-large``.
  - Run the command below:
  ```
  pip install -r requirements.txt
  ```
  - Install REx, and **change ``REx/rex/tasks`` with ``tasks`` in this repo**:
  ```
  git clone https://github.com/Spico197/REx.git
  cd REx
  pip install -e .
  ```

## Quick Start
### Pretrained Model weights & Datasets
Download pretrained model from [Downton/LDNet_Pretrain](https://huggingface.co/Downton/LDNet_Pretrain) and put the folder under ``LDNet_outputs`` folder. Download datasets from [Mirror](https://github.com/Spico197/Mirror/tree/main) and put them under ``resources`` folder. Zero-shot datasets are in [Spico/Mirror](https://huggingface.co/datasets/Spico/Mirror/tree/main/ent), put them under ``resources/Mirror`` folder. Download MIE datasets (Twitter-2015, Twitter-2017, MNRE) and transform them:
```
python data/txt2json.py ./ data/newInst/ T F F
```

### Evaluation
```
# main tasks
bash scripts/single_task_wPTAllExcluded_wInstruction/run.sh

# Multi-span and N-ary extraction
bash scripts/single_task_wPTAllExcluded_wInstruction/run_new_tasks.sh

# Few-shot
bash scripts/single_task_wPTAllExcluded_wInstruction/fewshot.sh

# Zero-shot
python -m src.eval

# Ablation
bash scripts/ablation.sh

# MIE
bash MIE.sh
```

### Training

```
rex train -m src.task -dc conf/Pretrain_ld.yaml
```
