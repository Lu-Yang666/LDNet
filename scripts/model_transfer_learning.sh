kd=true
bs=4
# NER
mp=LDNet_outputs/wPT_Ent_CoNLL03/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_Ent_CoNLL03/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Ent_CoNLL03_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_Ent_CoNLL03_mtl data_dir=resources/Mirror/uie/ent/conll03/remove_instruction

mp=LDNet_outputs/wPT_Ent_ACE04/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_Ent_ACE04/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace04.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Ent_ACE04_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace04.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_Ent_ACE04_mtl data_dir=resources/Mirror/uie/ent/ace04/remove_instruction

mp=LDNet_outputs/wPT_Ent_ACE05/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_Ent_ACE05/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace05.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Ent_ACE05_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace05.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_Ent_ACE05_mtl data_dir=resources/Mirror/uie/ent/ace05/remove_instruction

# RE
mp=LDNet_outputs/wPT_Rel_ACE05/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_Rel_ACE05/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_ace05.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Rel_ACE05_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_ace05.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_Rel_ACE05_mtl resources/Mirror/uie/rel/ace05-rel/remove_instruction

mp=LDNet_outputs/wPT_Rel_CoNLL04/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_Rel_CoNLL04/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Rel_CoNLL04_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_Rel_CoNLL04_mtl data_dir=resources/Mirror/uie/rel/conll04/remove_instruction

mp=LDNet_outputs/wPT_Rel_SciERC/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_Rel_SciERC/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Rel_SciERC_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_Rel_SciERC_mtl data_dir=resources/Mirror/uie/rel/scierc/remove_instruction

mp=LDNet_outputs/wPT_Rel_NYT/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_Rel_NYT/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Rel_NYT_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_Rel_NYT_mtl data_dir=resources/Mirror/uie/rel/nyt/remove_instruction

# EE
mp=LDNet_outputs/wPT_Event_ACE05/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_Event_ACE05/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Event_ACE05_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_Event_ACE05_mtl data_dir=resources/Mirror/uie/event/ace05-evt/remove_instruction

mp=LDNet_outputs/wPT_Event_CASIE/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_Event_CASIE/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_Event_CASIE_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_Event_CASIE_mtl data_dir=resources/Mirror/uie/event/casie/remove_instruction

# ABSA
mp=LDNet_outputs/wPT_ABSA_14res/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_ABSA_14res/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_ABSA_14res_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_ABSA_14res_mtl data_dir=resources/Mirror/uie/absa/14res/remove_instruction

mp=LDNet_outputs/wPT_ABSA_16res/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_ABSA_16res/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_16res.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_ABSA_16res_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_16res.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_ABSA_16res_mtl data_dir=resources/Mirror/uie/absa/16res/remove_instruction

mp=LDNet_outputs/wPT_ABSA_14lap/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_ABSA_14lap/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14lap.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_ABSA_14lap_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14lap.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_ABSA_14lap_mtl data_dir=resources/Mirror/uie/absa/14lap/remove_instruction

mp=LDNet_outputs/wPT_ABSA_15res/ckpt/SchemaGuidedInstructBertModel.best.pth
kd_file=LDNet_outputs/wPT_ABSA_15res/cache/logits.pt
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_15res.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=wPT_ABSA_15res_mtl 
torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_15res.yaml -a base_model_path=$mp kd=$kd kd_file=$kd_file train_batch_size=$bs eval_batch_size=$bs task_name=woInst_ABSA_15res_mtl data_dir=resources/Mirror/uie/absa/15res/remove_instruction