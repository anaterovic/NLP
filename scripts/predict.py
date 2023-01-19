# This file could contain a function that takes a model and a sentence and returns the predicted label.
# You could structure it the same way as the train or the evaluate scripts.

# parse args: --model_path --sentence --entity1 -entity2


import argparse
import torch
import pandas as pd

def predict(model_path: str, sentence: str, entity1: str, entity2: str):
    # load model
    model = torch.load(model_path)

    df = pd.DataFrame()
    df['sentence'] = sentence
    
    # predict
    pred = model.forward(df)

    # print
    print(pred)



def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-sm", "--saved_model", type=str, required=True, help="Path to model"
    )
    parser.add_argument(
        "-s", "--sentence", type=str, required=True, help="sentence"
    )
    parser.add_argument(
        "-e1", "--entity1", type=str, required=True, help="first entity"
    )
    parser.add_argument(
        "-e2", "--entity2", type=str, required=True, help="first entity"
    )
    # parser.add_argument(
    #     "-m", "--milestone", type=int, choices=[1, 2, 3], help="Milestone model to predict with"
    # )

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    model_path = args.saved_model
    sentence = args.sentence
    entity1 = args.entity1
    entity2 = args.entity2
    # milestone = args.milestone

    predict(model_path, sentence, entity1, entity2)




