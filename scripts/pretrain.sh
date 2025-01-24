# export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"

torchrun --nnodes=1 --nproc_per_node=8 -m rex.cmds.train -m src.task -dc conf/Pretrain_ld.yaml