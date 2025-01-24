rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/cadec.yaml -a task_name=DisNER_CADEC
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/hyperred.yaml -a task_name=HyperRel_HyperRED
