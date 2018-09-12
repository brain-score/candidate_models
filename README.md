# Brain-Score: Which Artificial Neural Network is most Brain-Like?
##### Neural net-specific framework

Candidate models to evaluate on brain measurements, i.e. neural and behavioral recordings.
Brain recordings are packaged [here](https://github.com/dicarlolab/brain-score).


## Quick setup

Recommended for most users. Use as a library.

```
# PyTorch -- current version cannot be installed from pip. Use the following for Python3.6-cpu or conda.
pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
pip install torchvision
# Brain-Score and Candidate-Models (this framework)
pip install --process-dependency-links git+https://github.com/dicarlolab/brain-score
pip install --process-dependency-links git+https://github.com/dicarlolab/candidate_models
```

During first-time use, ImageNet validation images (9.8 GB) will be downloaded, so give it a couple of minutes.

To contribute code to this framework, see the [Development Setup](#development-setup).


### Troubleshooting
###### Could not find a version that satisfies the requirement brain-score
pip has trouble when dependency links are private repositories (as is the case now for brain-score).
To circumvent, install brain-score by hand before installing candidate_models:
`pip install --process-dependency-links git+https://github.com/dicarlolab/brain-score`.

###### Could not find a version that satisfies the requirement tensorflow
TensorFlow doesn't always catch up with newer Python versions.
For instance, if you have Python 3.7 (check with `python -V`), TensorFlow might only work up to Python 3.6.
If you're using conda, it usually installs the very newest version of Python.
To fix, downgrade python: `conda install python=3.6`.

###### Failed to build pytorch
The current Pytorch version cannot be installed via pip.
Instead, you need to build pytorch from their provided wheel.
Check [the website](https://pytorch.org/) for installation instructions, right now they are (e.g. for Linux, Python 3.6, no CUDA):
`pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl &&
pip install torchvision`.
Or just use conda, e.g., for CPU: `conda install pytorch-cpu torchvision-cpu -c pytorch`


## Usage
```bash
PYTHONPATH=. python candidate_models --model alexnet
```

See the [examples](examples/) for more elaborate examples.


## Development setup

Only necessary if you plan to change code.

1. Clone the Git repository to wherever you keep repositories:
    * `cd ~/dev`
    * `git clone git@github.com:dicarlolab/candidate_models.git`
3. Create and activate a Conda environment with relevant packages:
    * `conda env create -f environment.yml`
    * `conda activate candidate-models`
