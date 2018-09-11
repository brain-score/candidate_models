# Brain-Score: Which Artificial Neural Network is most Brain-Like?
##### Neural net-specific framework

Candidate models to evaluate on brain measurements, i.e. neural and behavioral recordings.
Brain recordings are packaged [here](https://github.com/dicarlolab/brain-score).


## Quick setup

Recommended for most users. Use as a library.

`pip install git+https://github.com/dicarlolab/candidate_models`

You will also need to download ImageNet validation data to initialize the principal components.
Once you have done that, find the line [here](candidate_models/models/implementations/__init__.py) that points to `/braintree/data2/active/users/qbilius/datasets/imagenet2012.hdf5` and replace it with your downloaded ImageNet.
Or alternatively, give us a couple of days and this step will work automatically. :)

To contribute code to this framework, see the [Development Setup](#development-setup).


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
