bs=12
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/cadec.yaml -a base_model_path=null task_name=woPT_woInst_DisNER_CADEC data_dir=resources/Mirror/new_abilities_v2/cadec/new/remove_instruction train_batch_size=$bs eval_batch_size=$bs
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/hyperred.yaml -a base_model_path=null task_name=woPT_woInst_HyperRel_HyperRED data_dir=resources/Mirror/new_abilities_v2/HyperRED/new/remove_instruction train_batch_size=$bs eval_batch_size=$bs
