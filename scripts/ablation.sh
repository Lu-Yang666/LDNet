
ablation(){
    for droprate in {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}; do
        mp=LDNet_outputs/LDNet_Pretrain/ckpt/SchemaGuidedInstructBertModel.best.pth
        bs=100
        torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a lddrop=true droprate=$droprate base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs fewshot=true num_epochs=100 "task_name=wPT_ABSA_14res_ld_${droprate}" 
        torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a lddrop=true droprate=$droprate base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs fewshot=true num_epochs=100 "task_name=wPT_woInst_ABSA_14res_ld_${droprate}" data_dir=resources/Mirror/uie/absa/14res/remove_instruction
        torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a lddrop=true droprate=$droprate base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs fewshot=true num_epochs=100 "task_name=wPT_Ent_CoNLL03_ld_${droprate}" 
        bs=60
        torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a lddrop=true droprate=$droprate base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs fewshot=true num_epochs=100 "task_name=wPT_woInst_Event_ACE05_ld_${droprate}" data_dir=resources/Mirror/uie/event/ace05-evt/remove_instruction
        torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a lddrop=true droprate=$droprate base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs fewshot=true num_epochs=100 "task_name=wPT_Event_ACE05_ld_${droprate}" 
        bs=100
        torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a lddrop=true droprate=$droprate base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs fewshot=true num_epochs=100 "task_name=wPT_woInst_Ent_CoNLL03_ld_${droprate}" data_dir=resources/Mirror/uie/ent/conll03/remove_instruction
        bs=80
        torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a lddrop=true droprate=$droprate base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs fewshot=true num_epochs=100 "task_name=wPT_Rel_NYT_ld_${droprate}" 
        torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a lddrop=true droprate=$droprate base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs fewshot=true num_epochs=100 "task_name=wPT_woInst_Rel_NYT_ld_${droprate}" data_dir=resources/Mirror/uie/rel/nyt/remove_instruction
    done
}
ablation