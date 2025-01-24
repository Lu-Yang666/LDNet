# NER
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a task_name=woInst_Ent_CoNLL03 data_dir=resources/Mirror/uie/ent/conll03/remove_instruction
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace04.yaml -a task_name=woInst_Ent_ACE04 data_dir=resources/Mirror/uie/ent/ace04/remove_instruction
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace05.yaml -a task_name=woInst_Ent_ACE05 data_dir=resources/Mirror/uie/ent/ace05/remove_instruction

# RE
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_ace05.yaml -a task_name=woInst_Rel_ACE05 resources/Mirror/uie/rel/ace05-rel/remove_instruction
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a task_name=woInst_Rel_CoNLL04 data_dir=resources/Mirror/uie/rel/conll04/remove_instruction
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a task_name=woInst_Rel_SciERC data_dir=resources/Mirror/uie/rel/scierc/remove_instruction
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a task_name=woInst_Rel_NYT data_dir=resources/Mirror/uie/rel/nyt/remove_instruction

# EE
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a task_name=woInst_Event_ACE05 data_dir=resources/Mirror/uie/event/ace05-evt/remove_instruction
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a task_name=woInst_Event_CASIE data_dir=resources/Mirror/uie/event/casie/remove_instruction

# ABSA
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a task_name=woInst_ABSA_14res data_dir=resources/Mirror/uie/absa/14res/remove_instruction
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_16res.yaml -a task_name=woInst_ABSA_16res data_dir=resources/Mirror/uie/absa/16res/remove_instruction
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14lap.yaml -a task_name=woInst_ABSA_14lap data_dir=resources/Mirror/uie/absa/14lap/remove_instruction
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_15res.yaml -a task_name=woInst_ABSA_15res data_dir=resources/Mirror/uie/absa/15res/remove_instruction