import argparse

from tuwnlpie import logger
from tuwnlpie.milestone1.model import SimpleNBClassifier
from tuwnlpie.milestone1.utils import (
    calculate_tp_fp_fn,
    read_docs_from_csv,
    split_train_dev_test,
)

from tuwnlpie.milestone2.model import BoWClassifier
from tuwnlpie.milestone2.utils import IMDBDataset, Trainer


def evaluate_milestone1(test_data, saved_model, split=False):
    model = SimpleNBClassifier()
    model.load_model(saved_model)
    docs = read_docs_from_csv(test_data)
    test_docs = None

    if split:
        _, _, test_docs = split_train_dev_test(docs)
    else:
        test_docs = docs

    y_true = []
    y_pred = []
    for doc in test_docs:
        pred = model.predict_label(doc[0])
        y_true.append(doc[1])
        y_pred.append(pred)
        logger.info(f"Predicted: {pred}, True: {doc[1]}")

    tp, fp, fn, precision, recall, fscore = calculate_tp_fp_fn(y_true, y_pred)

    print("Statistics:")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {fscore}")

    return


def evaluate_milestone2(test_data, saved_model, split=False):
    logger.info("Loading data...")
    dataset = IMDBDataset(test_data)
    model = BoWClassifier(dataset.OUT_DIM, dataset.VOCAB_SIZE)
    model.load_model(saved_model)
    trainer = Trainer(dataset=dataset, model=model)

    logger.info("Evaluating...")
    test_loss, test_prec, test_rec, test_fscore = trainer.evaluate(
        dataset.test_iterator
    )

    print("Statistics:")
    print(f"Loss: {test_loss}")
    print(f"Precision: {test_prec}")
    print(f"Recall: {test_rec}")
    print(f"F1-Score: {test_fscore}")

    return


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
