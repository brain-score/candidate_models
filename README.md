[![Build Status](https://travis-ci.com/brain-score/candidate_models.svg?token=vqt7d2yhhpLGwHsiTZvT&branch=master)](https://travis-ci.com/brain-score/candidate_models)

# Candidate Models for Brain-Score: Which Artificial Neural Network is most Brain-Like?

Candidate models to evaluate on brain measurements, i.e. neural and behavioral recordings.
Brain recordings are packaged in [Brain-Score](https://github.com/dicarlolab/brain-score).


## Quick setup

```
pip install "candidate_models @ git+https://github.com/brain-score/candidate_models"
```

The above command will not install ML frameworks such as Pytorch, TensorFlow or Keras.

You can install them yourself or using the following commands (in a conda environment):
* Pytorch: `conda install pytorch torchvision -c pytorch`
* Keras: `conda install keras`
* TensorFlow: `conda install tensorflow`. To use the predefined TensorFlow models, you will have to install the [TF-slim library](https://github.com/qbilius/models/tree/master/research/slim).
See [here](#installing-the-tf-slim-image-models-library) for quick instructions.


## Usage
```bash
PYTHONPATH=. python candidate_models --model alexnet
```

During first-time use, ImageNet validation images (9.8 GB) will be downloaded, so give it a couple of minutes.

See the [examples](examples/) for more elaborate examples.


### Environment variables
Environment variables are prefixed with `CM_` for this framework. 
Environment variables from [brain-score](https://github.com/brain-score/brain-score) 
and [model-tools](https://github.com/brain-score/model-tools) might also be useful.

| Variable               | Description                                                  |
|------------------------|--------------------------------------------------------------|
| CM_HOME                | path to framework root                                       |
| CM_TSLIM_WEIGHTS_DIR   | path to stored weights for TensorFlow/research/slim models   |
| MT_IMAGENET_PATH       | path to ImageNet file containing the validation image set    |
| RESULTCACHING_HOME     | directory to cache results (benchmark ceilings) in, `~/.result_caching` by default (see https://github.com/mschrimpf/result_caching) |


## Installing the TF-slim image models library

TensorFlow-slim does unfortunately not provide an actual pip-installable library here, instead we have to download the code and make it available.

```bash
git clone https://github.com/qbilius/models/ tf-models
export PYTHONPATH="$PYTHONPATH:$(pwd)/tf-models/research/slim"
# verify
python -c "from nets import cifarnet; mynet = cifarnet.cifarnet"
```

Alternatively, you can also move/symlink these packages to your site-packages.


## Troubleshooting
<details>
<summary>Could not find a version that satisfies the requirement brain-score</summary>

pip has trouble when dependency links are private repositories (as is the case now for brain-score).
To circumvent, install brain-score by hand before installing candidate_models:
`pip install --process-dependency-links git+https://github.com/dicarlolab/brain-score`.
</details>

<details>
<summary>Could not find a version that satisfies the requirement tensorflow</summary>

TensorFlow doesn't always catch up with newer Python versions.
For instance, if you have Python 3.7 (check with `python -V`), TensorFlow might only work up to Python 3.6.
If you're using conda, it usually installs the very newest version of Python.
To fix, downgrade python: `conda install python=3.6`.
</details>

<details>
<summary>Failed to build pytorch</summary>

Some versions of PyTorch cannot be installed via pip (e.g. CPU).
Instead, you need to build pytorch from their provided wheel.
Check [the website](https://pytorch.org/) for installation instructions, right now they are (e.g. for Linux, Python 3.6, no CUDA):
`pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl &&
pip install torchvision`.
Or just use conda, e.g., for CPU: `conda install pytorch-cpu torchvision-cpu -c pytorch`
</details>

<details>
<summary>No module named `nets` / `preprocessing`</summary>
You probably haven't installed TensorFlow/research/slim.
Follow the instructions [here](https://github.com/tensorflow/models/tree/master/research/slim#Install).
</details>

<details>
<summary>ImportError: cannot import name '_obtain_input_shape'</summary>

keras_squeezenet unfortunately does not run with keras > 2.2.0.
To fix, `pip install keras==2.2.0`.
</details>

<details>
<summary>tensorflow.python.framework.errors_impl.FailedPreconditionError: Error while reading resource variable</summary>

If this happened when running a keras model, your tensorflow and keras versions are probably incompatible.
See the setup.py for which versions are supported.
</details>

<details>
<summary>Restoring from checkpoint failed. (...) Assign requires shapes of both tensors to match.</summary>

Most likely your passed image_size does not match up with the image size the model expects (e.g. inception_v{3,4} expect 299 insead of 224).
Either let the framework infer what image_size the model needs (run without `--image_size`) or set the correct image_size yourself.
</details>

<details>
<summary>MobileNet weight loading failed.</summary>

Error message e.g. `Assign requires shapes of both tensors to match. lhs shape= [1,1,240,960] rhs shape= [1,1,240,1280]`.

There is an error in the MobileNet implementation which causes the multiplier to not be applied properly:
the number of channels sometimes go beyond what they ought to be (e.g. for the last layer).
The [line in question](https://github.com/tensorflow/models/blob/628b970a3d7c59a3b65220e24972f9987e879bca/research/slim/nets/mobilenet/mobilenet.py#L250) needs to be prefixed with a conditional:
```
if i != len(conv_defs['spec']) - 1 or multiplier >= 1:
    opdef.multiplier_func(params, multiplier)
```
This is already done in [@qbilius' fork of tensorflow/models](https://github.com/qbilius/models).
</details>


<details>
<summary>Installation error due to version mismatch after re-submission.</summary>

Error message e.g. 
```
ERROR: Cannot install brain-score and candidate-models==0.1.0 because these package versions have conflicting dependencies.
The conflict is caused by:
    candidate-models 0.1.0 depends on pandas==0.25.3
    brainio-base 0.1.0 depends on pandas>=1.2.0
```

This can happen when re-submitting a model because the underlying submission.zip might point to versions that were okay at the time, but are in conflict after updates to the brain-score framework. For instance, old versions of candidate-models specified pandas==0.25.3 which was removed in newer versions and leads to old versions being incompatible with newer specifications of pandas in BrainIO.

The best solution is to re-submit a zip file without those version conflicts. Ideally submissions should avoid specifying any versions themselves as much as possible to prevent this error.
We have also been updating the zip files internally on the server, but this is not a long-term solution.
</details>

