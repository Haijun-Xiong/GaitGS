CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgs/gaitgs.yaml --phase train --log_to_file
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgs/gaitgs128.yaml --phase train --log_to_file
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 opengait/main.py --cfgs ./configs/gaitgs/gaitgs_OUMVLP.yaml --phase train --log_to_file