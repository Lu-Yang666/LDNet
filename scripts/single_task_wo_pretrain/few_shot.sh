#!/bin/bash

export bs=28

uie_fewshot() {
    for seed in $(seq 1 10); do
        for shot in {1,5,10}; do
            echo "seed=$seed, shot=$shot"
            echo "=== ent_conll03 ==="
            rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=woPT_Ent_CoNLL03_seed${seed}_${shot}shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed${seed}/${shot}shot" base_model_path=null train_batch_size=$bs eval_batch_size=$bs 
            echo "=== rel_conll04 ==="
            rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=woPT_Rel_CoNLL04_seed${seed}_${shot}shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed${seed}/${shot}shot" base_model_path=null train_batch_size=$bs eval_batch_size=$bs
            echo "=== event_ace05 ==="
            rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=woPT_Event_ACE05_seed${seed}_${shot}shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed${seed}/${shot}shot" base_model_path=null train_batch_size=$bs eval_batch_size=$bs 
            echo "=== absa_16res ==="
            rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=woPT_ABSA_16res_seed${seed}_${shot}shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed${seed}/${shot}shot" base_model_path=null train_batch_size=$bs eval_batch_size=$bs 
        done
    done
}

uie_fewshot
