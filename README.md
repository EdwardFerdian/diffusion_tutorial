# diffusion_tutorial

Create a new environment with Python 3.10 and install the package in editable mode:

```bash
conda create -n tutorial python=3.10
conda activate tutorial
pip install -e .
```

### 2D diffusion model

#### Training

1. Put the data in the `data` folder
2. Run the training script

```bash
# Input your data and output dir. The channels is the number of channels in the input data
python trainer_2d.py --data-dir data --output-dir models --channels 3

```

#### Inference

1. Run the inference script

```bash
python sampler_2d.py --model-dir models
```
