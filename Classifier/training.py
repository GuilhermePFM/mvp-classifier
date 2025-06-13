from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from typing import Tuple
from feature_engineering import TARGET_VARIABLE


def split_train_test(complete_dataset:pd.DataFrame, test_size:float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the complete dataset into training and testing sets.
    Args:
        complete_dataset (pd.DataFrame): The complete dataset containing features and target variable.
    Returns:
        tuple: A tuple containing the training features (train), testing features (test),
    """

    train, test = train_test_split(complete_dataset, test_size=.3, random_state=7, stratify=complete_dataset[TARGET_VARIABLE], shuffle=True)

    return train, test