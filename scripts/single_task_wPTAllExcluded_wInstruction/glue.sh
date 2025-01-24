rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/cola.yaml -a task_name=Cls_CoLA
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/qqp.yaml -a task_name=Cls_QQP
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/mnli.yaml -a task_name=Cls_MNLI
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/sst-2.yaml -a task_name=Cls_SST2
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/rte.yaml -a task_name=Cls_RTE
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/qnli.yaml -a task_name=Cls_QNLI
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/mrpc.yaml -a task_name=Cls_MRPC
rex train -m src.task -dc conf/LDNet-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/squad_v2.yaml -a task_name=Span_SQuADv2
