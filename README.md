# Brain-Score: Which Artificial Neural Network is most Brain-Like?

Candidate models to evaluate on brain measurements, i.e. neural and behavioral recordings.
Brain recordings are packaged [here](https://github.com/dicarlolab/brain-score).

## Installation
Install dependencies with conda:
```bash
conda env create -f environment.yml
conda activate candidate-models
```

## Usage
```bash
PYTHONPATH=. python candidate_models --model alexnet
```
