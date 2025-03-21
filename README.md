# RoboPack: Learning Tactile-Informed Dynamics Models for Dense Packing

RoboPack is a framework that integrates tactile-informed state estimation, dynamics prediction, and planning for manipulating objects with ***unknown*** physical properties. It extends previous work [RoboCraft](http://hxu.rocks/robocraft/) and [RoboCook](https://hshi74.github.io/robocook/) by incorporating tactile-informed physical state estimation to handle uncertainties in object properties, such as unknown mass distribution or compliance.

Packing objects with varying deformability using one initial visual observation and dense tactile feedback:

https://github.com/user-attachments/assets/8e1f2d81-69ba-4f98-86da-8d37c6b64dc2

**RoboPack: [Website](https://robo-pack.github.io/) |  [Paper](https://arxiv.org/abs/2407.01418)**

If you find this codebase useful for your research, please consider citing: 
```
@article{ai2024robopack,
  title={RoboPack: Learning Tactile-Informed Dynamics Models for Dense Packing},
  author={Bo Ai and Stephen Tian and Haochen Shi and Yixuan Wang and Cheston Tan and Yunzhu Li and Jiajun Wu},
  journal={Robotics: Science and Systems (RSS)},
  year={2024},
  url={https://www.roboticsproceedings.org/rss20/p130.pdf},
}
```
and the previous work that this codebase is built upon. 

## Environment
Dependencies have been exported to `requirement.txt`. The most important is to have compatible versions for `torch` and `torch_geometric`. 

## Sample Dataset
We provide a small sample dataset to help get started with running the pipeline. You can download it [here](https://drive.google.com/file/d/1KS9Zyp4Z9K7R0F13zmgMpsP21Wa589C2/view?usp=sharing).  
After downloading, please unzip it in the project root folder:
```angular2html
cd robopack
unzip data.zip
```
The example commands below assume that the data directory `robopack/data` has already been set up.

## Learning Tactile Auto-Encoder
First, navigate to `dynamics`
```angular2html
cd dynamics
```

Below is an example command for training a tactile encoder on the box-pushing dataset:
```angular2html
python train_tactile_encoder.py --config model_configs/estimator_predictor_tac_packing_seq25.json
```
In practice, we train the encoder on an aggregated dataset, which is then shared across tasks.

To generate visualizations from a pretrained autoencoder for inspection, here is an example of testing a checkpoint:
```angular2html
 python train_tactile_encoder.py --config model_configs/estimator_predictor_tac_boxes.json --test /home/albert/github/robopack-public/dynamics/pretrained_ae/v24_5to5_epoch=101-step=70482_corrected.ckpt
```
The generated visualization videos will be saved in `ae_visualizations`.

## Dynamics Learning
To run a minimal example of dynamics learning, run one of the following the following
```angular2html
python train_dynamics.py --config model_configs/estimator_predictor_tac_boxes.json  # box pushing task
python train_dynamics.py --config model_configs/estimator_predictor_tac_packing_seq25.json  # dense packing task 
```
