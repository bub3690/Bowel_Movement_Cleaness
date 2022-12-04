
python baseline_train.py --batch-size 32 --epochs 40 --lr 0.0001 --model sub_2stage --sublabel label --wandb True --seed 1012 --tag 2stage 1012 --num-workers 20 --project-name BMC_vision_classification --pretrained-chkpt ./checkpoint/sub_1stage_get_label_512_segment_False_augment__seed_1012.pt
python baseline_train.py --batch-size 32 --epochs 40 --lr 0.0001 --model sub_2stage --sublabel label --wandb True --seed 1013 --tag 2stage 1013 --num-workers 20 --project-name BMC_vision_classification --pretrained-chkpt ./checkpoint/sub_1stage_get_label_512_segment_False_augment__seed_1013.pt