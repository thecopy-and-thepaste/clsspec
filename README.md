# Readme

# clsspec

## Overview

**clsspec** (classify-species) deals with the problem of binary classification of species names in written texts. 

The complete method relies on the end-to-end library, [CONABIO_ML_TEXT](https://bitbucket.org/conabio_cmd/conabio_ml_text/src/master/), which also takes on the [CONABIO_ML](https://bitbucket.org/conabio_cmd/conabio_ml/src/master/).

**Note:** Both of the [CONABIO_ML](https://bitbucket.org/conabio_cmd/conabio_ml/src/master/) and [CONABIO_ML_TEXT](https://bitbucket.org/conabio_cmd/conabio_ml_text/src/master/) libraries are in early stages of development. So, the usage of **dev branches** is assumed.

## Assumptions

The main premise is set on the dataset. Which is we can build a set of ngrams that contains species names, and then, contrast it with common knowledge ngrams. 

So, we gather the species names from [the GBIF](https://api.gbif.org) site to build a domain dataset (d-dataset) (you can download the one used in the example from [HERE](https://tctp-datasets.s3.us-south.cloud-object-storage.appdomain.cloud/species.txt)). Then, we process common texts files to build an out-of-domain dataset (ood-dataset), in the example we use abstracts from [the PLOS](https://plos.org/) site using the query `subject:life sciences` (the dataset is available to download [HERE](https://tctp-datasets.s3.us-south.cloud-object-storage.appdomain.cloud/clsspec_text_files.zip)).

For the model, we train a Recurrent Neural Network using nchar tokens  to classify a ngram into the `species/not-species` classes. The methodology is heavily ground on this [Andrej Karpathy's post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

## Workflow

It comprises 2 stages: the dataset building/merging and the model training/evaluation. Whose are briefly described in the following:

- Dataset building (eda.ipynb)

We merge and process both **ood** and **d** datasets to produce a new dataset at ngram level. We also produce some statistics like number of classes and length of the data.

- **Training/prediction (clsspec.ipynb/pipeline.py)**

We use the [CONABIO_ML_TEXT](https://bitbucket.org/conabio_cmd/conabio_ml_text/src/master/) library to draw the substeps of the methodology. Such as, dataset loading, further processing, and model training/evaluation.

## Requirements

Python 3.7.7+

[CONABIO_ML_TEXT](https://bitbucket.org/conabio_cmd/conabio_ml_text/src/master/)

[docker](https://www.docker.com/) (optionally docker-compose)

## Environment

The fastest (and the recommended) way to set up the environment is using docker/docker-compose.

- **docker-compose**

You can simply start your environment using the `docker-compose.yml` file located in root directory and build/start with:

```python
docker-compose build
docker-compose start
```

This method works with the image of the [CONABIO_ML_TEXT](https://bitbucket.org/conabio_cmd/conabio_ml_text/src/master/) library and only shares the directory `code`with the container.

Note: This methods DOES NOT map the GPU with the container.

- **docker**

If you want to take advantage of an environment with GPU ready, you can create the image manually and share the code folder with the following commands:

```python
docker image build -t TAG:v0 conabio_ml_text/images/tf2
docker run -it --gpus NUM_GPUS --name TAG -d -v host_path/envs:/lib/code_environment/code -p 9000:8888 -p 9001:6006 TAG:v0 bash

Where:
	host_path: Path where the clsspec project is located
	TAG: string to recognize the container
	-p (option): to map ports of jupyter notebook and tensorboard.
```

## Configuration

We recommend you first visit `eda/clsspec` notebooks, to catch some insights of the workflow. Both of them are pretty straight forward. Nonetheless, if you choose to perform it vía pipeline (using `pipeline.py`) you can get extra resources of the process performed. 

You always first need to exec `eda.ipynb`, to create the dataset. Then, run the pipeline using the config file located in the `configs` folder, with:

```python
python code/pipeline.py -c code/configs/config.json
Where:
	c is the config json file
```

The `config.json`contains the parameters used in the experiment. A template is provided:

```python
{
    # Dataset built on eda.ipynb
    "dataset": "code/dataset/dataset.csv", 
	# model layers, we use 1 input, 1 embedding, 1 lstm and the final classifier
    "layers": {
        "input": {
            "T": 49
        },
        "embedding": {
            "V": 11660,
            "D": 100
        },
        "lstm": {
            "M": 20
        },
        "dense": {
            "K": 2
        }
    },
	# hyperparams
    "params": {
        "initial_learning_rate": 0.0001,
        "decay_steps": 200,
        "batch_size": 16,
        "epochs": 1
    }
}
```

Having finished the pipeline execution you will have a 'package' with some results and assets, that corresponds to every step of the pipeline. In the folder path defined by the variable `results_path`

The main results are stored in the `results_path/evaluate/results.json` file.