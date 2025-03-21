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

## Dynamics Learning
To run a minimal example of dynamics learning, first download a sample dataset [here](https://drive.google.com/file/d/1KS9Zyp4Z9K7R0F13zmgMpsP21Wa589C2/view?usp=sharing), then run the following
```angular2html
cd robopack
unzip data.zip
cd dynamics 
python train_dynamics.py --config model_configs/estimator_predictor_tac_boxes.json
```
This will load a sample dataset for the non-prehensile box-pushing task.

