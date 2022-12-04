import argparse

from tuwnlpie import logger
from tuwnlpie.milestone1.model import NBClassifier
from tuwnlpie.milestone1.utils import read_food_disease_csv, split_data, calculate_tp_fp_fn

from tuwnlpie.milestone2.model import Trainer, TorchModel
from tuwnlpie.milestone2.utils import read_crowd_truth_csv

from pytorch_lightning.core.lightning import LightningModule


import numpy as np
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')


def evaluate_milestone1(test_data, saved_model, split=False):
    model = NBClassifier()
    model.load_model(saved_model)
    docs = read_food_disease_csv(test_data)
    test_docs = None

    if split:
        _, test_docs = split_data(docs)
    else:
        test_docs = docs

    y_true = []
    y_pred = []
    for _, row in test_docs.iterrows():
        pred = model.predict_label(row['sentence'], row['food_entity'], row['disease_entity'])
        y_true.append(row[['is_cause', 'is_treat']])
        y_pred.append(pred)
        logger.info(f"Predicted: {pred}, True: {row[['is_cause', 'is_treat']]}")

    print(classification_report(
        np.array(y_true).squeeze().astype(int),
        np.array(y_pred).squeeze().astype(int),
        target_names=['is_cause', 'is_treat']))


def evaluate_milestone2(test_data, saved_model, split=False):
    model = TorchModel.load_from_checkpoint("tuwnlpie/milestone2/lightning_logs/version_0/checkpoints/epoch=1-step=12.ckpt")
    trainer = Trainer()
    docs = read_crowd_truth_csv(test_data)
    test_docs = None

    if split:
        _, test_docs = split_data(docs)
    else:
        test_docs = docs

    trainer.test(model, test_docs)



def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-t", "--test-data", type=str, required=True, help="Path to test data"
    )
    parser.add_argument(
        "-sm", "--saved-model", type=str, required=True, help="Path to saved model"
    )
    parser.add_argument(
        "-sp", "--split", default=False, action="store_true", help="Split data"
    )
    parser.add_argument(
        "-m", "--milestone", type=int, choices=[1, 2], help="Milestone to evaluate"
    )

    return parser.parse_args()


if "__main__" == __name__:
    args = get_args()

    test_data = args.test_data
    model = args.saved_model
    split = args.split
    milestone = args.milestone

    if milestone == 1:
        evaluate_milestone1(test_data, model, split=split)
    elif milestone == 2:
        evaluate_milestone2(test_data, model, split=split)
