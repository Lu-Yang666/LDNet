# main tasks w/o pretrain

# NER
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace04.yaml -a base_model_path=null task_name=woPT_Ent_ACE04
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace05.yaml -a base_model_path=null task_name=woPT_Ent_ACE05
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a base_model_path=null task_name=woPT_Ent_CoNLL03

# RE
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_ace05.yaml -a base_model_path=null task_name=woPT_Rel_ACE05
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a base_model_path=null task_name=woPT_Rel_CoNLL04_gp 
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a base_model_path=null task_name=woPT_Rel_NYT_gp 
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a base_model_path=null task_name=woPT_Rel_SciERC_gp 

# EE
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a base_model_path=null task_name=woPT_Event_ACE05
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a base_model_path=null task_name=woPT_Event_CASIE

# ABSA
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a base_model_path=null task_name=woPT_ABSA_14res
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14lap.yaml -a base_model_path=null task_name=woPT_ABSA_14lap
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_15res.yaml -a base_model_path=null task_name=woPT_ABSA_15res
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_16res.yaml -a base_model_path=null task_name=woPT_ABSA_16res
