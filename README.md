
# Relation extraction

For this project we had to research and implement various classifiers for prediction of relationship between entities. It was done as part of Natural Language Processing and Information Extraction course. </br>
Main documents are located in the docs folder.

## Install and Quick Start

First create a new conda environment with python 3.10 and activate it:

```bash
conda create -n tuwnlpie python=3.10
conda activate tuwnlpie
```

Then install this repository as a package, the `-e` flag installs the package in editable mode, so you can make changes to the code and they will be reflected in the package.

```bash
pip install -e .
```

All the requirements should be specified in the `setup.py` file with the needed versions. If you are not able to specify everything there
you can describe the additional steps here, e.g.:

Install `black` library for code formatting:
```bash
pip install black
```

Install `pytest` library for testing:
```bash
pip install pytest
```

## Run Milestone 1

__Training__:

To train a model on the FoodDisease dataset and then save object to a file, you can run a command:

```bash
python ./scripts/train.py -t ./data/food_disease.csv -s -sp ./models/model_milestone1.pkl -m1
```

__Evaluation__:

To evaluate the model on the dataset with a trained model, you can run a command:

```bash
python scripts/evaluate.py -t data/food_disease.csv -sm data/bayes_model.pkl -sp -m 1
```

## Run Milestone 2

__Training__:

To train the neural network on the IMDB dataset and then save the weights to a file, you can run a command:

```bash
python ./scripts/train.py -t ./data/crowd_truth_combined.csv -sdp -s -sp ./models/model_milestone2.pt -m2
```

__Evaluation__:
    
To evaluate the model on the dataset with the trained weights, you can run a command (you can also provide a pretrained model, so if someone wants to evaluate your model, they can do it without training it):

```bash
# python scripts/evaluate.py -t data/imdb_dataset_sample.csv -sm data/bow_model.pt -sp -m 2
```

## Running the tests

For testing we use the `pytest` package (you should have it installed with the command provided above). You can run the tests with the following command:

```bash
pytest
```

## Code formatting
To convey a consistent coding style across the project, we advise using a code formatter to format your code.
For code formatting we use the `black` package (you should have it installed with the command provided above). You can format your code with the following command:

```bash
black .
```
