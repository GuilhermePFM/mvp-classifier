from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import pandas as pd
from feature_engineering import TEXT_FEATURES


MODEL_NAME = "distilbert-base-uncased"


def tokenized_pytorch_tensors(
        df: pd.DataFrame,
        column_list: list
    ) -> Dataset:
    """
    Tokenizes text in a pandas DataFrame and converts it to a Dataset of PyTorch tensors.
    Args:
        df (pd.DataFrame): DataFrame containing text data to be tokenized.
        column_list (list): List of columns to be included in the output Dataset.
    Returns:
        Dataset: A Dataset object containing tokenized text as PyTorch tensors.
        """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    transformers_dataset = Dataset.from_pandas(df)

    def tokenize(model_inputs_batch: Dataset) -> Dataset:
        return tokenizer(
            model_inputs_batch[TEXT_FEATURES[0]],
            padding=True,
            max_length=120,
            truncation=True,
        )

    tokenized_dataset = transformers_dataset.map(
        tokenize,
        batched=True,
        batch_size=128,
    )

    tokenized_dataset.set_format(
        "torch",
        columns=column_list
    )
    
    columns_to_remove = set(tokenized_dataset.column_names) - set(column_list)

    tokenized_dataset = tokenized_dataset.remove_columns(list(columns_to_remove))

    return tokenized_dataset

import numpy as np
def hidden_state_from_text_inputs(df) -> pd.DataFrame:

    def extract_hidden_states(batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)

        inputs = {
            k: v.to(device)
            for k, v in batch.items()
            if k in tokenizer.model_input_names
        }

        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state
            # get the CLS token, which is the first one
            # [:, 0] gives us a row for each batch with the first column of 768 for each
            return {"cls_hidden_state": np.asarray(last_hidden_state[:, 0].cpu().numpy())}

    cls_dataset = df.map(extract_hidden_states, batched=True, batch_size=128)
    cls_dataset.set_format(type="pandas")

    return pd.DataFrame(
        cls_dataset["cls_hidden_state"].to_list(),
        columns=[f"feature_{n}" for n in range(1, 769)],#TODO: check number of features
    )