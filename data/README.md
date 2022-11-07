## Datasets
- This folder should contain all the data you are going to use in your project. 
- Instead of puting the data directly to GitHub, you should provide a link to the data in the README.md file and instruct the user how to download the data.
- This is a good practice since usually we don't want to version control the data, but we want to version control the code that processes the data.

### The IMDB dataset
- The IMDB dataset is a dataset of movie reviews.
- It contains 50,000 reviews, each of which is labeled as positive or negative.
- We will use a sample of this dataset to make the computation faster.

Download it using command line:
```bash
wget https://owncloud.tuwien.ac.at/index.php/s/C2EXAQBlMLHvpHv/download -O imdb_dataset_sample.csv
```

We also provide pretrained models.

To download the pretrained naive bayes model, run the following command:

```bash
wget https://owncloud.tuwien.ac.at/index.php/s/L0z1sXXgWFLloY4/download -O bayes_model.tsv
```

To download the pretrained neural network model, run the following command:

```bash
wget https://owncloud.tuwien.ac.at/index.php/s/hEJEq7vvN2ErtUj/download -O bow_model.pt
```

You could also run the following script to download everything:

```bash
bash download.sh
```