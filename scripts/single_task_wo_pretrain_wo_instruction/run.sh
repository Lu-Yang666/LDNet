# main tasks w/o pretrain w/o Inst.

# NER
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a base_model_path=null data_dir=resources/Mirror/uie/ent/conll03/remove_instruction task_name=woPT_woInst_Ent_CoNLL03
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace04.yaml -a base_model_path=null data_dir=resources/Mirror/uie/ent/ace04/remove_instruction task_name=woPT_woInst_Ent_ACE04
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace05.yaml -a base_model_path=null data_dir=resources/Mirror/uie/ent/ace05/remove_instruction task_name=woPT_woInst_Ent_ACE05

# RE
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_ace05.yaml -a base_model_path=null data_dir=resources/Mirror/uie/rel/ace05-rel/remove_instruction task_name=woPT_woInst_Rel_ACE05
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a base_model_path=null data_dir=resources/Mirror/uie/rel/conll04/remove_instruction task_name=Mirror_SingleTask_woPT_woInst_Rel_CoNLL04
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a base_model_path=null data_dir=resources/Mirror/uie/rel/scierc/remove_instruction task_name=Mirror_SingleTask_woPT_woInst_Rel_SciERC
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a base_model_path=null data_dir=resources/Mirror/uie/rel/nyt/remove_instruction task_name=Mirror_SingleTask_woPT_woInst_Rel_NYT

# EE
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a base_model_path=null data_dir=resources/Mirror/uie/event/ace05-evt/remove_instruction task_name=woPT_woInst_Event_ACE05
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a base_model_path=null data_dir=resources/Mirror/uie/event/casie/remove_instruction task_name=woPT_woInst_Event_CASIE

# ABSA
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a base_model_path=null data_dir=resources/Mirror/uie/absa/14res/remove_instruction task_name=woPT_woInst_Ent_ABSA_14res
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_16res.yaml -a base_model_path=null data_dir=resources/Mirror/uie/absa/16res/remove_instruction task_name=woPT_woInst_ABSA_16res
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14lap.yaml -a base_model_path=null data_dir=resources/Mirror/uie/absa/14lap/remove_instruction task_name=woPT_woInst_ABSA_14lap
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_15res.yaml -a base_model_path=null data_dir=resources/Mirror/uie/absa/15res/remove_instruction task_name=woPT_woInst_ABSA_15res
