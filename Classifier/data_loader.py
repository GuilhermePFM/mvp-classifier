from pathlib import Path
import pandas as pd
from feature_engineering import TARGET_VARIABLE

DATASET_NAME = 'database_classified.parquet'


def remove_class_with_few_samples(complete_dataset: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    """ Remove classes with fewer samples than the specified minimum.       
    Args:
        complete_dataset (pd.DataFrame): The dataset to be filtered.
        min_samples (int): The minimum number of samples required for each class.
    Returns:
        pd.DataFrame: The filtered dataset with classes having fewer samples than min_samples removed.
    """
    samples_per_class = pd.DataFrame(complete_dataset.groupby(TARGET_VARIABLE[0])[TARGET_VARIABLE[0]].count())

    classes_bellow_treshold = samples_per_class[samples_per_class[TARGET_VARIABLE[0]] <= min_samples]

    classes_to_replace_list = classes_bellow_treshold.index.to_list()

    # Replace classes with fewer samples than the threshold with 'Other'
    complete_dataset[TARGET_VARIABLE[0]] = complete_dataset[TARGET_VARIABLE[0]].where(~complete_dataset[TARGET_VARIABLE[0]].isin(classes_to_replace_list), 'Other')

    return complete_dataset 

def treat_text(complete_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Treat the text in the dataset.
    Args:
        complete_dataset (pd.DataFrame): The dataset to be treated.
    Returns:
        complete_dataset (pd.DataFrame): Dataset with treated text
    """
    # treat text
    complete_dataset['Descrição'] = complete_dataset['Descrição'].str.lower()
    complete_dataset['Descrição'] = complete_dataset['Descrição'].fillna("")

    return complete_dataset

def treat_dataset(complete_dataset: pd.DataFrame):
    """
    Treat the dataset to prepare for pre-processing
    Args:
        complete_dataset (pd.DataFrame): The dataset to be treated.
    Returns:
        complete_dataset (pd.DataFrame): Dataset treated
    """
    # treat text
    complete_dataset = treat_text(complete_dataset)

    # for the classification models, the number of samples for each class must be greater than 2
    complete_dataset = remove_class_with_few_samples(complete_dataset, min_samples = 5)

    return complete_dataset


def load_raw_dataset(root_dir:Path):
    """
    Load the dataset.
    Args:
        root_dir (Path): The path of the dataset to be treated.
    Returns:
        complete_dataset (pd.DataFrame): Complete dataset
    """

    # Load the dataset
    complete_dataset = pd.read_parquet(root_dir / DATASET_NAME)

    return complete_dataset


def load_treated_dataset(root_dir:Path):
    """
    Load the dataset.
    Args:
        root_dir (Path): The path of the dataset to be treated.
    Returns:
        complete_dataset (pd.DataFrame): Complete dataset treated
    """
    df = load_raw_dataset(root_dir)
    return treat_dataset(df)