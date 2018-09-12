# Brain-Score: Which Artificial Neural Network is most Brain-Like?
##### Neural net-specific framework

Candidate models to evaluate on brain measurements, i.e. neural and behavioral recordings.
Brain recordings are packaged [here](https://github.com/dicarlolab/brain-score).


## Quick setup

Recommended for most users. Use as a library.

```
pip install git+https://github.com/dicarlolab/brain-score
pip install git+https://github.com/dicarlolab/candidate_models
```

During first-time use, ImageNet validation images (9.8 GB) will be downloaded, so give it a couple of minutes.

To contribute code to this framework, see the [Development Setup](#development-setup).


### Troubleshooting
**Could not find a version that satisfies the requirement brain-score**
pip has trouble when dependency links are private repositories (as is the case now for brain-score).
To circumvent, install brain-score by hand before installing candidate_models: `pip install git+https://github.com/dicarlolab/brain-score`.


**Could not find a version that satisfies the requirement tensorflow**
TensorFlow doesn't always catch up with newer Python versions.
For instance, if you have Python 3.7 (check with `python -V`), TensorFlow might only work up to Python 3.6.
If you're using conda, it usually installs the very newest version of Python.
To fix, downgrade python: `conda install python=3.6`.


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
