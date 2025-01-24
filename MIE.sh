python run.py --data_path='data/newInst/merged' --bce_mean=1 --num_epochs=20 --eval_begin_epoch=10 --batch_size=32 --save_path=ldnet_mie  --do_train=1 --use_images=1 --use_inst=1 --use_ldnet=1

# winst
python run.py --data_path='data/newInst/NER/data/twitter2015/' --bce_mean=1 --num_epochs=20 --eval_begin_epoch=10 --batch_size=32 --save_path=ldnet_twitter_15 --load_path=ldnet_mie --do_train=1 --use_inst=1 --use_images=1 --use_ldnet=1
python run.py --data_path='data/newInst/NER/data/twitter2017/' --bce_mean=1 --num_epochs=20 --eval_begin_epoch=10 --batch_size=32 --save_path=ldnet_twitter_17 --load_path=ldnet_mie --do_train=1 --use_inst=1 --use_images=1 --use_ldnet=1
python run.py --data_path='data/newInst/RE/data/mre/' --bce_mean=1 --num_epochs=20 --eval_begin_epoch=10 --batch_size=32 --save_path=ldnet_mre --load_path=ldnet_mie --do_train=1 --use_inst=1 --use_images=1 --use_ldnet=1

# woinst
python run.py --data_path='data/newInst/NER/data/twitter2017/' --bce_mean=1 --num_epochs=20 --eval_begin_epoch=10 --batch_size=32 --save_path=ldnet_twitter_17_woinst --load_path=ldnet_mie --do_train=1 --use_images=1 --use_ldnet=1
python run.py --data_path='data/newInst/NER/data/twitter2015/' --bce_mean=1 --num_epochs=20 --eval_begin_epoch=10 --batch_size=32 --save_path=ldnet_twitter_15_woinst --load_path=ldnet_mie --do_train=1 --use_images=1 --use_ldnet=1
python run.py --data_path='data/newInst/RE/data/mre/' --bce_mean=1 --num_epochs=20 --eval_begin_epoch=10 --batch_size=32 --save_path=ldnet_mre_woinst --load_path=ldnet_mie --do_train=1 --use_images=1 --use_ldnet=1