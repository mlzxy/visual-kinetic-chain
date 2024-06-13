# Scaling Manipulation Learning with Visual Kinematic Chain Prediction

[Arixv Paper](http://arxiv.org/abs/2406.07837)   [Website](https://mlzxy.github.io/visual-kinetic-chain/)

![](docs/demo.gif)

This repo contains code for the VKT implementation, download datasets from [box drive](https://rutgers.box.com/s/yuv1ey8twbnbqbj2r86dfee74vz5pn4g), extract 
and change the dataset path in [configs/vkt/default.py](configs/vkt/default.py).

Then, install `torch` and `torchvision`, and other dependencies.

```bash
pip install -r ./requirements.txt
```


Next, run the training command: 

```bash
accelerate launch  --config_file ./accelerate.config.yaml  train.py  --config  ./configs/vkt/default.py   --wandb YOUR_WANDB_PROJECT_ID   --module.data.num_workers 10
```

Note:

1. The above dataset does not include UR5, sawyer and language table, even thought language table is shown in the above gif.
2. This code includes the visual kinematics chain forecasting part, the remaining code will be uploaded later. 
