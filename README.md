# SRLSTM
States Refinement LSTM\
This is the code for 
[SR-LSTM: State Refinement for LSTM towards Pedestrian Trajectory Prediction. CVPR2019](https://arxiv.org/pdf/1903.02793.pdf).

## Environment
The code is tested on Ubuntu 16.04, Python 3.5.4, numpy 1.13.3, pytorch 1.0.1.post2.

## Train
The Default settings are to train on ETH-univ. Data cache and models will be in the subdirectory "./savedata/0/".
```
python .../SRLSTM/train.py
```
Configuration files are also created after the first run, arguments could be modified through configuration files. \
Priority: command line \> configuration files \> default values in script

The datasets are selected on arguments '--test_set'. Five datasets in ETH/UCY are corresponding to the value of \[0,1,2,3,4\]. 

This command is to train model for ETH-hotel and save cache files in '/Your/save/directory/1'.
```
python .../SRLSTM/train.py --test_set 1 --save_base_dir '/Your/save/directory'
```
You can set your model name by "--train_model" and model type by "--model". 

Detailed arguments description is given in train.py.

## Test
```
python .../SRLSTM/test.py --test_set X --load_model XXX
```
Test example models are given in ./savedata/X/testmodel/testmodel_XXX.tar\
To test on UCY-univ, using 
```
python .../SRLSTM/test.py --test_set 4 --load_model 324 --batch_around_ped 64
```
To test on your own models, use your train.py and change the arguments of  '--phase', '--train_model','--load_model'
to 'test','YourModelName','YourModelEpoch'.

## Citation
If you find this code useful, please cite us as 
```
@inproceedings{zhang2019srlstm,
  title={SR-LSTM: State Refinement for LSTM towards Pedestrian Trajectory Prediction},
  author={Zhang, Pu and Ouyang, Wanli and Zhang, Pengfei and Xue, Jianru and Zheng, Nanning},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

