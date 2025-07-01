from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    Normalizer,
)
from tokenization import tokenized_pytorch_tensors
from sklearn.preprocessing import FunctionTransformer
from tokenization import hidden_state_from_text_inputs
from feature_engineering import add_keyword_hints, add_date_features
from sklearn.pipeline import FeatureUnion
from feature_engineering import NUMERIC_FEATURES, DATE_FEATURES, TEXT_FEATURES


VERBOSE = False


def description_transformer():
    return make_pipeline(
        FunctionTransformer(tokenized_pytorch_tensors, kw_args={"column_list": ["input_ids", "attention_mask"]}),
        FunctionTransformer(hidden_state_from_text_inputs),
        verbose=VERBOSE
    )


def new_feature_hints_transformer():
    return make_pipeline(
        FunctionTransformer(add_keyword_hints),
        Normalizer(),
        verbose=VERBOSE
    )


def text_transformer(pytorch):
    if pytorch:
        estimators = [
            ('description', description_transformer()), 
            ('keyword_hints', new_feature_hints_transformer())]
    else:
        estimators = [ 
            ('keyword_hints', new_feature_hints_transformer())]

    combined = FeatureUnion(estimators)
    return combined


def date_transformer():
    return make_pipeline( FunctionTransformer(add_date_features),
                          Normalizer(),
                          verbose=VERBOSE
    )


def get_preprocessing_transformer( VERBOSE=False, pytorch=True) -> ColumnTransformer:
    preprocessor = make_column_transformer(
                    ( StandardScaler(), NUMERIC_FEATURES),
                    ( date_transformer(), DATE_FEATURES),
                    ( text_transformer(pytorch), TEXT_FEATURES),
        remainder='passthrough', verbose=VERBOSE)
    return preprocessor 