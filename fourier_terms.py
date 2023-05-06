import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def _calculate_fourier_terms(
    seasonal_cycle: np.ndarray, max_cycle: int, n_fourier_terms: int
):
    """Calculates Fourier Terms given the seasonal cycle and max_cycle"""
    sin_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    cos_X = np.empty((len(seasonal_cycle), n_fourier_terms), dtype="float64")
    for i in range(1, n_fourier_terms + 1):
        sin_X[:, i - 1] = np.sin((2 * np.pi * seasonal_cycle * i) / max_cycle)
        cos_X[:, i - 1] = np.cos((2 * np.pi * seasonal_cycle * i) / max_cycle)
    return np.hstack([sin_X, cos_X])


def add_fourier_features(
    df: pd.DataFrame,
    column_to_encode: str,
    max_value: Optional[int] = None,
    n_fourier_terms: int = 1,
    use_32_bit: bool = False,
) -> Tuple[pd.DataFrame, List]:
    """Adds Fourier Terms for the specified seasonal cycle column, like month, week, hour, etc.
    Args:
        df (pd.DataFrame): The dataframe which has the seasonal cycles which has to be encoded
        column_to_encode (str): The column name which has the seasonal cycle
        max_value (int): The maximum value the seasonal cycle can attain. for eg. for month, max_value is 12.
            If not given, it will be inferred from the data, but if the data does not have at least a
            single full cycle, the inferred max value will not be appropriate. Defaults to None
        n_fourier_terms (int): Number of fourier terms to be added. Defaults to 1
        use_32_bit (bool, optional): Flag to use float32 or int32 to reduce memory. Defaults to False.
    Raises:
        warnings.warn: Raises a warning if max_value is None
    Returns:
        [Tuple[pd.DataFrame, List]]: Returns a tuple of the new dataframe and a list of features which were added
    """
    assert (
        column_to_encode in df.columns
    ), "`column_to_encode` should be a valid column name in the dataframe"
    assert is_numeric_dtype(
        df[column_to_encode]
    ), "`column_to_encode` should have numeric values."
    if max_value is None:
        max_value = df[column_to_encode].max()
        raise warnings.warn(
            "Inferring max cycle as {} from the data. This may not be accuracte if data is less than a single seasonal cycle."
        )
    fourier_features = _calculate_fourier_terms(
        df[column_to_encode].astype(int).values,
        max_cycle=max_value,
        n_fourier_terms=n_fourier_terms,
    )
    feature_names = [
        f"{column_to_encode}_sin_{i}" for i in range(1, n_fourier_terms + 1)
    ] + [f"{column_to_encode}_cos_{i}" for i in range(1, n_fourier_terms + 1)]
    df[feature_names] = fourier_features
    if use_32_bit:
        df[feature_names] = df[feature_names].astype("float32")
    return df, feature_names