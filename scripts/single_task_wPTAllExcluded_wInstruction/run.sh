mp=LDNet_outputs/LDNet_Pretrain/ckpt/SchemaGuidedInstructBertModel.best.pth

# NER
bs=200
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Ent_CoNLL03 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace04.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Ent_ACE04 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace05.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Ent_ACE05 

# RE
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_ace05.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Rel_ACE05 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Rel_CoNLL04 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Rel_SciERC 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Rel_NYT 

# EE
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Event_ACE05 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Event_CASIE 

# ABSA
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_ABSA_14res 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_16res.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_ABSA_16res 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14lap.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_ABSA_14lap 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_15res.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=wPT_ABSA_15res 