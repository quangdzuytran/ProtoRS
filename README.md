# ProtoRS
## Example
Training:
```
python main.py --batch_size 128 --epochs 110 --log_dir runs/protors_cub --dataset CUB-200-2011 --lr 0.01 --lr_block 0.001 --lr_net 1e-05 --lr_rule 0.001 --num_features 256 --num_prototypes 101 --structure 101 --net resnet50_inat --freeze_epochs 15 --soft_epochs 15 --projection_cycle 5 --milestones 60,70,80,90,100
```
Model explanation:
```
python explain_global.py --state_dict_dir_model runs/protors_cub/checkpoints/best_test_model --log_dir runs/protors_cub --batch_size 128
```
Explaining individual predictions (must run `explain_global.py` beforehand):
```
python explain_local.py --model runs/protors_cub/explained_model --log_dir runs/protors_cub/explained_model --dataset CUB-200-2011 --sample_dir data/CUB_200_2011/dataset/test_full/017.Cardinal/Cardinal_0014_17389.jpg
```
