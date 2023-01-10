# train centernet
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_centernet.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_addr 127.0.0.2 --master_port 20002 train_centernet.py

# train tttfnet
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 train_ttfnet.py
#python -m torch.distributed.launch --master_port=29500 --nproc_per_node=1 train_ttfnet.py

# prune ttfnet with rmnet or repvgg
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=29500 --nproc_per_node=4 slim_ttfnet.py --sparse_rate 0.01
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port=29501 --nproc_per_node=4 slim_ttfnet.py --pruned_finetune # --prune_ratio 0.6

# prune ttfnet with dp
# CUDA_VISIBLE_DEVICES=1 python slim_ttfnet_dp.py --per_node_batch_size 64 --num_workers 16 --sparse_rate 0.001
# CUDA_VISIBLE_DEVICES=1 python slim_ttfnet_dp.py --per_node_batch_size 64 --num_workers 16 --pruned_finetune  # --prune_ratio 0.6

# CUDA_VISIBLE_DEVICES=1 python slim_ttfnet_dp.py --per_node_batch_size 64 --num_workers 16 --sparse_rate 0.001
# CUDA_VISIBLE_DEVICES=1 python slim_ttfnet_dp.py --per_node_batch_size 64 --num_workers 16 --pruned_finetune --finetune_idx 1 --sparse_rate 0.001
# CUDA_VISIBLE_DEVICES=1 python slim_ttfnet_dp.py --per_node_batch_size 64 --num_workers 16 --pruned_finetune --finetune_idx 2
