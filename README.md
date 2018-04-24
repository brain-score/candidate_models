# Neurality of deep neural nets compared to cortex
Evaluate how well your model predicts neural data 
(retrieved from [mkgu](https://github.com/dicarlolab/mkgu): macaque monkey ventral stream).

## Installation
Install dependencies with conda:
```bash
conda env create -f environment.yml
conda activate neurality
```

## Usage
```bash
PYTHONPATH=. python neurality --model vgg16 \
  --layers block1_pool block2_pool block3_pool block4_pool block5_pool fc1 fc2
```
