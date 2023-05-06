import pandas as pd
from collections import defaultdict
from tqdm.autonotebook import tqdm


def compact_to_expanded(
    df, timeseries_col, static_cols, time_varying_cols, ts_identifier
):
    """
    Converting compact to expanded ts
    """
    def preprocess_expanded(x):
        ### Fill missing dates with NaN ###
        # Create a date range from  start
        dr = pd.date_range(
            start=x["start_timestamp"],
            periods=len(x["energy_consumption"]),
            freq=x["frequency"],
        )
        df_columns = defaultdict(list)
        df_columns["timestamp"] = dr
        for col in [ts_identifier, timeseries_col] + static_cols + time_varying_cols:
            df_columns[col] = x[col]
        return pd.DataFrame(df_columns)

    all_series = []
    for i in tqdm(range(len(df))):
        all_series.append(preprocess_expanded(df.iloc[i]))
    df = pd.concat(all_series)
    del all_series
    return df