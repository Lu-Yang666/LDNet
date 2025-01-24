mp=null
bs=12
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/cadec.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=woPT_DisNER_CADEC
bs=4
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/hyperred.yaml -a base_model_path=$mp train_batch_size=$bs eval_batch_size=$bs task_name=woPT_HyperRel_HyperRED
