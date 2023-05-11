import pandas as pd
from typing import List, Tuple

def get_df(split: str = "train") -> pd.DataFrame:
    dataset_path: str = f"datasets/nocola_ungrammatical_{split}.txt"
    data: List[Tuple[str, str]] = []

    with open(dataset_path, 'r') as file:
        for line in file:
            line = line.strip()
            sentence_parts = line.split('\t')
            if len(sentence_parts) == 2:
                sentence_text, class_label = sentence_parts
                data.append((sentence_text, class_label))

    df: pd.DataFrame = pd.DataFrame(data, columns=['text', 'label'])

    invalid_labels: List[str] = ['X', 'FL']
    df = df[~df['label'].isin(invalid_labels)]

    return df
