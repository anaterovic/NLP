import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split

from tuwnlpie import logger
from tuwnlpie.milestone1.model import NBClassifier
from tuwnlpie.milestone1.utils import read_food_disease_csv, split_data

from tuwnlpie.milestone2.model import TorchModel
from tuwnlpie.milestone2.utils import TorchTrainer, read_and_prepare_data, TorchDataset, split_data_set, \
                                        length_longest_sentence, encodeX


def train_milestone1(train_data=Path('..', 'data', 'food_disease.csv'), use_sdp=False, save=False, save_path=None):
    print("## Reading in Data ##")
    docs = read_food_disease_csv(train_data)

    print("## Split ##")
    X = docs[['food_entity', 'disease_entity', 'sentence']]
    X = X['sentence']
    y = docs[['is_cause', 'is_treat']]
    X_train, X_test, y_train, y_test = split_data(X, y,  test_size=0.8, random_state=1)

    print("## Create the model ##")
    model = NBClassifier(use_sdp=use_sdp)

    print("## Starting Training ##")
    model.train(X_train, y_train)
    print("## model trained sucessfully ##")
    
    if save:
        print(f"## saving model to {save_path} ##")
        model.save_model(save_path)
        logger.info(f"Saved model to {save_path}")
    return


def train_milestone2(train_data, use_sdp=True, save=False, save_path=None):

    feature_col = 'tokens_lemma' #['term1', 'term2', 'sentence', 'tokens', 'tokens_stem', 'tokens_lemma']
    label_cols = ['is_cause', 'is_treat']

    # readIn
    print("## Reading in Data ##")
    data_frame = read_and_prepare_data(train_data, shall_sdp=True)
    data_frame = data_frame[['tokens_lemma', 'tokens', 'is_cause', 'is_treat']]

    print(data_frame.columns.values)
    X = data_frame[['tokens_lemma', 'tokens']]
    y = data_frame[['is_cause', 'is_treat']]

    X = encodeX(X)
    y = y.to_numpy()

    print(X[4])
    return

    # X = X['tokens_lemma'] # overcome with encode

    return
    print("## Split ##")
    X_train, X_test, y_train, y_test = split_data_set(X, y,  test_size=0.8, random_state=1)
    X_test, X_val, y_test, y_val= split_data_set(X_test, y_test, test_size=0.5, random_state=1) 


    print("## Creating Data-Loaders ##")
    # Data Loaders
    X_train = TorchDataset(X_train, y_train)
    X_test = TorchDataset(X_test, y_test)
    X_val = TorchDataset(X_val, y_val)

    dataloaders = { 
        'train': X_train.get_dataloader(batch_size=256, shuffle=True), 
        'test': X_test.get_dataloader(batch_size=128, shuffle=False), 
        'val': X_val.get_dataloader(batch_size=128, shuffle=False)
    }
    
    print("## Creating Model ##")
    # model
    model = TorchModel()
    # trainer
    trainer = TorchTrainer(
        model, 
        'test', 
        "../tuwnlpie/milestone2/lightning_logs/version_0/checkpoints/" ,
        dataloaders,
        max_epochs=10
        )

    print("## Starting Training ##")
    the_trainer = trainer.run()

    if save:
        print(f"## Saving Model to {save_path} ##")
        torch.save(model, save_path)
    return


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--train-data", type=str, required=True, help="Path to training data"
    )
    parser.add_argument(
        "-sdp", "--shortest-dep-path", default=True, action="store_true", help="Use shortest dependency path tokens"
    )
    parser.add_argument(
        "-s", "--save", default=False, action="store_true", help="Save model"
    )
    parser.add_argument(
        "-sp", "--save-path", default=None, type=str, help="Path to save model"
    )
    parser.add_argument(
        "-m", "--milestone", type=int, choices=[1, 2], help="Milestone to train"
    )

    return parser.parse_args()


if "__main__" == __name__:
    args = get_args()

    train_data = args.train_data
    shortest_dep_path = args.shortest_dep_path
    model_save = args.save
    model_save_path = args.save_path
    milestone = args.milestone

    if milestone == 1:
        train_milestone1(train_data, use_sdp=shortest_dep_path, save=model_save, save_path=model_save_path)
    elif milestone == 2:
        train_milestone2(train_data, use_sdp=shortest_dep_path, save=model_save, save_path=model_save_path)
