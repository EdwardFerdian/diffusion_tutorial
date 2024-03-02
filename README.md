# diffusion_tutorial

Create a new environment with Python 3.10 and install the package in editable mode:

```bash
conda create -n tutorial python=3.10
conda activate tutorial
pip install -e .
```

## 2D diffusion model

#### Training

1. Put the data in the `data` folder
2. Run the training script

```bash
# Input your data and output dir. The channels is the number of channels in the input data
python trainer_2d.py --data-dir data --output-dir models [--channels 3]

```

#### Inference

1. Run the inference script

```bash
python sampler_2d.py --model-dir models
```

## 1D diffusion model

#### Training

1. Prepare your data in a HDF5 file with containing an _input_ column
2. Run the training script

```bash
# Input your data and output dir. The channels is the number of channels in the input data
python trainer_1d.py --input-file data.h5 --output-dir models [--seq-length 480]
```

#### Inference

1. Run the inference script

```bash
python sampler_1d.py --model-dir models
```

## References

This tutorial code is based on the implementation from:
https://github.com/lucidrains/denoising-diffusion-pytorch

Some modifications and fixes were made to the original code.
