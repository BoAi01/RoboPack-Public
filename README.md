# RoboPack: Learning Tactile-Informed Dynamics Models for Dense Packing

**RoboPack: [Website](https://robo-pack.github.io/) |  [Paper](https://arxiv.org/abs/2407.01418)**

If you use this code for your research, please cite:

```
@article{ai2024robopack,
  title={RoboPack: Learning Tactile-Informed Dynamics Models for Dense Packing},
  author={Bo Ai and Stephen Tian and Haochen Shi and Yixuan Wang and Cheston Tan and Yunzhu Li and Jiajun Wu},
  journal={Robotics: Science and Systems (RSS)},
  year={2024},
  url={https://www.roboticsproceedings.org/rss20/p130.pdf},
}
```


## Environment
Dependencies have been exported to `requirement.txt`. The most important is to have the compatible versions for `torch` and `torch_geometric`. 

## Dynamics Learning
To run a minimal example of dynamics learning, run the following
```angular2html
cd robopack-public/dynamics
python train_dynamics.py --config model_configs/estimator_predictor_tac_boxes.json
```
This will load a sample dataset for the non-prehensile box pushing task.

