# ProtoRS: Prototype Rule Set for Interpretable Image Classification

## Prerequisites
General:
- Python 3
- Pytorch 1.11
- CUDA

Python packages:
- numpy
- pandas
- opencv
- tqdm
- scipy
- matplotlib
- requests

The `env.yaml` contains the environment file to config the packages using Anaconda.

## Data
The code can be applied to the [CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/) dataset with 200 bird species
The folder `preprocess_data` contains python code to download, extract and preprocess these datasets. 

### Preprocessing CUB
1. Create a folder `./data/CUB_200_2011`
2. Download [ResNet50 pretrained on iNaturalist2017](https://drive.google.com/drive/folders/1yHme1iFQy-Lz_11yZJPlNd9bO_YPKlEU) (Filename on Google Drive: `BBN.iNaturalist2017.res50.180epoch.best_model.pth`) and place it in the folder `features/state_dicts`.
3. From the main ProtoTree folder, run `python preprocess_data/download_birds.py` 
4. From the main ProtoTree folder, run `python preprocess_data/cub.py` to create training and test sets

## Training & Generating Explanations

### Training
1. Create a folder `./runs/`
2. Run the command
```
python main.py --epochs 300 \
                --log_dir ./runs/protors_cub \
                --dataset CUB-200-2011 \
                --batch_size 128 \
                --lr 0.01 \
                --lr_net 1e-5 \
                --lr_block 0.001 \
                --lr_rule 0.001 \
                --num_features 256 \
                --num_prototypes 202 \
                --structure 202 \
                --net resnet50_inat \
                --freeze_epochs 20 \
                --soft_epochs 100 \
                --projection_start 30 \
                --projection_cycle 5 \
                --milestones 200,240,260,280,300 \
                --weight_decay 1e-3 \
                --binarize_threshold 0.8
```
(If you want to use the interlaced configuration, use the tag `--interlaced` and double the numbers in the `--structure` tag.)
3. Check your `--log_dir` to keep track of the training progress. This directory contains `log_epoch_overview.csv` which prints per epoch the test accuracy, mean training accuracy and the mean loss. File `log_train_epochs_losses.csv` prints the loss value and training accuracy per batch iteration. File `log.txt` logs additional info. 
4. To continue the training, use the tag `--resume` and specify the last epoch location by the tag `--state_dict_dir_model ./runs/<last log dir>/checkpoints/epoch_<last epoch>`

### Model explanation
1. To generate global explanation, use the command
```
python explain_global.py --log_dir ./runs/protors_cub \
                          --state_dict_dir_model runs/protors_cub/checkpoints/best_test_model \
                          --batch_size 128
```
2. The visualized prototypes are in the folder `./runs/protors_cub/explained_model/prototype_upsampling_results` and the extracted rule set is in the file `./runs/protors_cub/explained_model/ruleset.csv`

### Explaining individual predictions
1. If the folder `explained_model` does not exist, run the `explain_global.py` as above
2. Run the command
```
python explain_local.py --log_dir ./runs/protors_cub/explained_model \
                        --model ./runs/protors_cub/explained_model \
                        --dataset CUB-200-2011 \
                        --sample_dir data/CUB_200_2011/dataset/test_full/017.Cardinal/Cardinal_0014_17389.jpg
```
3. The visualized matched prototypes and the activated rules are in the folder `./runs/protors_cub/explained_model/local_explanations`

## Acknowledgments
This code is based on the implement of 
- ProtoTree which can be found at [github.com/M-Nauta/ProtoTree](https://github.com/M-Nauta/ProtoTree)
- RRL which can be found at [github.com/12wang3/rrl](https://github.com/12wang3/rrl)
